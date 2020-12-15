#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
import torchsummary

#%%
class Pyramid(nn.Module):
    def __init__(self, input_chans=12, output_chans=8,
                layer_filters=[64, 128, 128, 256],
                kernel_sizes=[256, 128, 64, 32, 16]):
        super().__init__()
        self.input_chans=input_chans
        self.output_chans=output_chans
        self.layer_filters = layer_filters
        self.kernel_sizes = kernel_sizes

        self.conv_layers = nn.ModuleList([])
        self.conv_layers += [nn.Conv1d(input_chans,
                                layer_filters[0],
                                kernel_size=self.kernel_sizes[0]).double()]
        for idx in range(1, len(self.layer_filters)):
            self.conv_layers += [nn.Conv1d(self.layer_filters[idx-1],
                                        self.layer_filters[idx],
                                        kernel_size=self.kernel_sizes[idx]).double()]
        self.conv_layers += [nn.Conv1d(self.layer_filters[-1],
                                    self.output_chans,
                                    kernel_size=self.kernel_sizes[-1]).double()]
    
    def calc_receptive_field(self):
        return np.sum(np.array(self.kernel_sizes)) - len(self.kernel_sizes)

    def forward(self, x):
        for i in range(len(self.conv_layers)-1):
            x = self.conv_layers[i](x)
            #x = F.tanh(x)
        x = self.conv_layers[-1](x)
        
        return x

class skip_model(nn.Module):
    def __init__(self, input_chans=12, output_chans=8,
                layer_filters=[64, 128, 256],
                kernel_sizes=[128, 128, 128]):
        super().__init__()
        self.input_chans = input_chans
        self.output_chans = output_chans
        self.layer_filters = layer_filters
        self.kernel_sizes = kernel_sizes

        self.layer_1 = nn.Conv1d(self.input_chans, 128, kernel_size=129, padding=64).double()
        self.layer_2 = nn.Conv1d(140, 128, kernel_size=129, padding=64).double()
        self.layer_3 = nn.Conv1d(268, 8, kernel_size=129, padding=64).double()
    
    def forward(self, x):
        #print(f'input {x.shape}')
        y = self.layer_1(x)
        y = F.tanh(y)
        #print(f'layer_1 {x.shape}')

        x = torch.cat([x, y], dim=1)

        y = F.tanh(x)
        y = self.layer_2(y)
        #print(f'layer_1 y {y.shape}')

        x = torch.cat([x, y], dim=1)

        x = F.tanh(x)
        x = self.layer_3(x)

        return x

#device = "cuda"
#model = skip_model().to(device)
#recep = model.calc_receptive_field()
#torchsummary.summary(model, (12, 1548))

"""
a = torch.ones((1, 12, 1548)).to(device)
p = model(a)
p.shape, recep, 1548 - recep"""
# %%
class Custom_Conv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, padding=0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv1d(in_chans, out_chans,
                            kernel_size=self.kernel_size,
                            padding=padding)
        #self.batch_norm = nn.BatchNorm1d(out_chans)
        
    def forward(self, x):
        x = self.conv(x)
        #x = self.batch_norm(x)
        x = x * x * x
        return x

class Bi_Conv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, padding=0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv_1 = nn.Conv1d(in_chans, out_chans,
                            kernel_size=self.kernel_size,
                            padding=padding)
        self.conv_2 = nn.Conv1d(in_chans, out_chans,
                            kernel_size=self.kernel_size,
                            padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_chans)
        
    def forward(self, x):
        x_p = x
        x_m = -x
        x_p = self.conv_1(x_p)
        x_p = F.relu(x_p)

        x_m = self.conv_2(x_m)
        x_m = F.relu(x_m)

        x = torch.sub(x_p, x_m)
        x = self.batch_norm(x)
        return x

class Simple_Custom_CNN(nn.Module):
    def __init__(self, input_chans=12, output_chans=8,
                layer_filters=[64, 128, 256],
                kernel_sizes=[128, 128, 128]):
        super().__init__()
        self.input_chans = input_chans
        self.output_chans = output_chans
        self.layer_filters = layer_filters
        self.kernel_sizes = kernel_sizes

        self.layer_1 = Bi_Conv(self.input_chans, 128, kernel_size=129, padding=64).double()
        self.layer_2 = Bi_Conv(140, 128, kernel_size=129, padding=64).double()
        self.layer_3 = nn.Conv1d(268, 8, kernel_size=129, padding=64).double()
    
    def forward(self, x):
        #print(f'input {x.shape}')
        y = self.layer_1(x)
        #print(f'layer_1 {x.shape}')

        x = torch.cat([x, y], dim=1)

        y = self.layer_2(x)
        #print(f'layer_1 y {y.shape}')

        x = torch.cat([x, y], dim=1)

        x = self.layer_3(x)

        return x

class Simple_CNN_sigmoid(nn.Module):
    def __init__(self, input_chans=12, output_chans=8,
                layer_filters=[64, 128, 256],
                kernel_sizes=[128, 128, 128]):
        super().__init__()
        self.input_chans = input_chans
        self.output_chans = output_chans
        self.layer_filters = layer_filters
        self.kernel_sizes = kernel_sizes

        self.layer_1 = Custom_Conv(self.input_chans, 128, kernel_size=129, padding=64).double()
        self.layer_2 = Custom_Conv(140, 128, kernel_size=129, padding=64).double()
        self.layer_3 = Custom_Conv(268, 8, kernel_size=129, padding=64).double()
    
    def forward(self, x):
        #print(f'input {x.shape}')
        y = self.layer_1(x)
        #print(f'layer_1 {x.shape}')

        x = torch.cat([x, y], dim=1)

        y = self.layer_2(x)
        #print(f'layer_1 y {y.shape}')

        x = torch.cat([x, y], dim=1)

        x = self.layer_3(x)

        return x