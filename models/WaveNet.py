#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import numpy as np
# %%
class time_distributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first
        
    def forward(self, x):
        # x should be (batch, time, freq) => ((batch * time), channel) reshape
        # (batch, channel, time) = 
        if len(x.size()) <= 2:
            return self.module(x)
        
        # merge batch and time
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1)) # (batch, timestep, freq)
        else:
            y = y.view(-1, x.size(1), y.size(-1)) #(timestep, batch, freq)
        return y
#%%
class WaveNet(nn.Module):
    def __init__(self,
                layer_size=8,
                stack_size=1,
                residual_channels = 24,
                skip_connections = 128,
                input_channels = 12,
                output_channels = 8,
                linear=True,
                device = "cpu",
                bias=False):
        super(WaveNet, self).__init__()
        #Hyper parameters
        self.device = device
        input_channel = input_channels
        self.output_channels = output_channels
        self.residual_channels = residual_channels
        self.linear = linear

        #Build Model
        self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)

        self.dilations = []
        self.dilated_queues = []


        self.causal_conv = CausalConv1d(in_channels=input_channel, out_channels=residual_channels)
        self.res_stack = ResidualStack(layer_size=layer_size, stack_size=stack_size, 
                                    res_channels=residual_channels, skip_channels=skip_connections)
        
        self.conv_1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_1_bn = nn.BatchNorm1d(128)
        self.conv_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_2_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        
        if not linear:
            self.fcs = nn.ModuleList(LastFC(128, 256) for _ in range(self.output_channels))
        else:
            self.fcs = nn.ModuleList(LastFC(128, 1) for _ in range(self.output_channels))
    
    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(layer_size)] * stack_size
        num_receptive_fields = np.sum(np.unique(np.array(layers)))
        
        return int(num_receptive_fields)
    
    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields
        #print("output size: ", output_size)
        self.check_input_size(x, output_size)

        return output_size
    
    def check_input_size(self, x, output_size):
        if output_size < 1:
            raise NameError('The data x is too short! The expected output size is {}'.format(output_size))
    
    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        output = x.transpose(1,2).contiguous()
        output_size = self.calc_output_size(output)
        
        output = self.causal_conv(output)
        
        skips = self.res_stack(output, output_size)
        skips = torch.sum(skips, dim=0) # (batch, chan, time)
        skips = self.conv_1(skips)
        skips = self.conv_1_bn(skips)
        skips = self.relu(skips)
        skips = self.conv_2(skips)
        skips = self.conv_2_bn(skips)
        skips = self.relu(skips) # (batch, channels, time)

        output = torch.tensor([]).to(self.device)
        for fc_layer in self.fcs:
            decision = fc_layer(skips).unsqueeze(-1)
            output = torch.cat([output, decision], dim=-1)
            #print("output.shape", output.shape)
        if not self.linear:
            return F.tanh(output)
        else:
            return F.tanh(output.contiguous().squeeze(1))

#%%
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(CausalConv1d, self).__init__()

        # To match size of input and output, set padding = 1
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            padding=1, stride=1, kernel_size=2, bias=bias)

    def forward(self, x):
        #To make size of output same as size of input, remove last value
        x = self.conv(x)[:,:,:-1] #(minibatch, channels, length) 
        return x

#%%
class DilatedConv1d(nn.Module):
    def __init__(self, channels, dilation=1, bias=False):
        super(DilatedConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels,
                            kernel_size=2, stride=1,
                            dilation=dilation, padding=0,
                            bias=bias)
        
    def forward(self, x):
        output = self.conv(x)
        return output

#%%
class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = DilatedConv1d(channels=res_channels, dilation=dilation, bias=bias)
        self.conv_res = nn.Conv1d(in_channels=res_channels, out_channels=res_channels, padding=0, stride=1, kernel_size=1)
        self.conv_skip = nn.Conv1d(in_channels=res_channels, out_channels=skip_channels, padding=0, stride=1, kernel_size=1)

        self.gated_tanh = nn.Tanh()
        self.gated_sig = nn.Sigmoid()
    
    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        """
        output = self.dilated_conv(x)

        # forward for gate
        gated_t = self.gated_tanh(output)
        gated_s = self.gated_sig(output)

        gated = gated_t * gated_s
        
        # forward for residual conv
        input_cut = x[:,:,-output.size(2):]
        res = self.conv_res(gated) + input_cut

        #forward for skip conv
        skip = self.conv_skip(gated)
        skip = skip[:,:,-skip_size:]

        return output, skip

#%%
class ResidualStack(nn.Module):
    def __init__(self, layer_size=10, stack_size=5, res_channels=32, skip_channels=1):
        """
        Stack ResidualBlock by layer and stack size
        """
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size
        self.res_channels = res_channels
        self.skip_channels = skip_channels

        self.res_blocks = self.stack_res_block(res_channels, skip_channels)
    
    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)
        
        if torch.cuda.is_available():
            block.cuda()
        
        return block

    def stack_res_block(self, res_channels, skip_channels):
        """
        Prepare dilated convolution blocks by layer and stack size
        """
        res_blocks = []
        dilations = self.build_dilations()

        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, dilation)
            res_blocks.append(block)
        
        return res_blocks
            
    
    def build_dilations(self):
        dilations = []
        # 5 = stack[layer 1, layer 2, layer 3, layer 4, layer 5]
        for s in range(self.stack_size):
            # 10 = layer[dilation=1, dilation=2,4,8,16,32,64,128,256,512]
            for l in range(self.layer_size):
                dilations.append(2 ** l)
        
        return dilations
    
    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            #output is the next input
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)
        
        return torch.stack(skip_connections)

#%%
class LastFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Last network of the wavenet
        :param channels: number of channels for input and output
        :return:
        """
        super().__init__()
        self.fc_1 = nn.Linear(in_features=128, 
                            out_features=512, bias=False)
        self.fc_1 = time_distributed(self.fc_1)

        self.fc_2 = nn.Linear(in_features=512, 
                            out_features=out_channels, bias=False)
        self.fc_2 = time_distributed(self.fc_2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fc = x.contiguous().transpose(1, 2) # (batch, time, chan)
        fc = self.fc_1(fc)
        fc = self.relu(fc)
        fc = self.fc_2(fc) # (batch, time, chan) 
        
        output = fc.contiguous().transpose(1,2) # (batch, chan, time)
        return output


# %%
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# %%
device ="cuda" if torch.cuda.is_available() else "cpu"
model = WaveNet(device=device).to(device)
x = torch.ones((1, 1500, 12)).to(device)
torchsummary.summary(model, (1500, 12))

# %%
y = model(x)
y.shape
# %%
"""