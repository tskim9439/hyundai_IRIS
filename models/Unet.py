#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
# %%
# Input size (batch, chan, freq, time) => (None, 12, 64, time)
class conv2d_block(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, strides=1, padding=1, activation='leaky_relu'):
        super().__init__()
        self.conv2d_layer = nn.Conv2d(in_chans,
                                out_chans,
                                kernel_size=kernel_size,
                                padding=padding)
        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU()
        self.normalization = nn.BatchNorm2d(out_chans)
    
    def forward(self, x):
        x = self.conv2d_layer(x)
        x = self.activation(x)
        x = self.normalization(x)
        return x
6
class dense_block(nn.Module):
    def __init__(self, in_chans, out_chans, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.in_chans = in_chans
        self.out_chans = out_chans

        conv_layers = [conv2d_block(in_chans, out_chans, kernel_size=3, strides=1, padding=1, activation='leaky_relu')]
        conv_layers += [conv2d_block(out_chans * i,
                            out_chans,
                            kernel_size=3,
                            strides=1,
                            padding=1,
                            activation='leaky_relu') for i in range(1, self.n_layers)]
        self.conv_layers = nn.ModuleList(conv_layers)
    
    def calculate_output_chans(self):
        return self.out_chans * self.n_layers

    def forward(self, x):
        x = self.conv_layers[0](x)
        for i in range(1, len(self.conv_layers)):
            y = self.conv_layers[i](x)
            x = torch.cat([x, y], dim=1)
        return x

class UNet(nn.Module):
    def __init__(self, input_chans=12, output_chans=8, n_layers=[3, 4, 4, 3], filters=[4,4,4,4]):
        super().__init__()
        # input (batch, chans, freq, time) => (12, 128, 128)
        self.input_chans = input_chans
        self.output_chans = output_chans
        self.n_layers = n_layers
        self.filters = filters

        #Define Encoder
        encoder = []
        decoder_inchans = []
        in_chans = self.input_chans
        for i, (n_filter, n_layer) in enumerate(zip(self.filters, self.n_layers)):
            encoder += [dense_block(in_chans=in_chans, out_chans=n_filter, n_layers=n_layer)]
            in_chans = encoder[-1].calculate_output_chans()
            decoder_inchans += [in_chans]

        self.encoder = nn.ModuleList(encoder)

        #Define Decoder
        decoder = []
        transpose_layers = []
        decoder_block_outchans = []
        decoder_inchans = decoder_inchans[::-1]
        decoder_filters = self.filters[::-1]
        decoder_layers = self.n_layers[::-1]
        previous_outchans = 0
        for i, (n_filter, n_layer) in enumerate(zip(decoder_filters, decoder_layers)):
            decoder += [dense_block(in_chans=decoder_inchans[i] + previous_outchans,
                                    out_chans=n_filter,
                                    n_layers=n_layer)]
            if i < len(decoder_filters) - 1:
                transpose_layers += [nn.ConvTranspose2d(
                    in_channels=decoder[-1].calculate_output_chans(),
                    out_channels=decoder[-1].calculate_output_chans(),
                    kernel_size=2,
                    stride=2
                    )]
                previous_outchans = decoder[-1].calculate_output_chans()
        self.decoder = nn.ModuleList(decoder)
        self.transpose_layers = nn.ModuleList(transpose_layers)
        
        self.decision = nn.Conv2d(in_channels=decoder[-1].calculate_output_chans(),
                                out_channels=self.output_chans,
                                kernel_size=3,
                                padding=1)

    def forward(self, x):
        stack = []
        for idx, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            #print(f'encode {idx+1} : {x.shape}')
            stack += [x]
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        stack = stack[::-1]
        x = self.decoder[0](stack[0])
        #print(f'x : {x.shape}')
        for idx, t_layer in enumerate(self.transpose_layers):
            y = t_layer(x)
            #print(f'y {y.shape}')
            #print(f'stack {stack[1 + idx].shape}')
            x = torch.cat([stack[1 + idx], y], dim=1)
            #print(f'TranposedConv2d : {x.shape}')
            x = self.decoder[1 + idx](x)
            #print(f'decoder {x.shape}')
        x = self.decision(x)
        x = F.tanh(x)

        return x

class UNet_magphase_split(nn.Module):
    def __init__(self, input_chans=12, output_chans=8, n_layers=[3, 4, 4, 3], filters=[4,4,4,4]):
        super().__init__()
        # input (batch, chans, freq, time) => (12, 128, 128)
        self.input_chans = input_chans
        self.output_chans = output_chans
        self.n_layers = n_layers
        self.filters = filters

        #Define Encoder
        encoder = []
        decoder_inchans = []
        in_chans = self.input_chans
        for i, (n_filter, n_layer) in enumerate(zip(self.filters, self.n_layers)):
            encoder += [dense_block(in_chans=in_chans, out_chans=n_filter, n_layers=n_layer)]
            in_chans = encoder[-1].calculate_output_chans()
            decoder_inchans += [in_chans]

        self.encoder = nn.ModuleList(encoder)

        #Define Decoder for magnitude
        decoder_mag = []
        decoder_phase = []
        transpose_layers_mag = []
        transpose_layers_phase = []
        decoder_block_outchans = []
        decoder_inchans = decoder_inchans[::-1]
        decoder_filters = self.filters[::-1]
        decoder_layers = self.n_layers[::-1]
        previous_outchans = 0
        for i, (n_filter, n_layer) in enumerate(zip(decoder_filters, decoder_layers)):
            decoder_mag += [dense_block(in_chans=decoder_inchans[i] + previous_outchans,
                                    out_chans=n_filter,
                                    n_layers=n_layer)]
            decoder_phase += [dense_block(in_chans=decoder_inchans[i] + previous_outchans,
                                    out_chans=n_filter,
                                    n_layers=n_layer)]
            if i < len(decoder_filters) - 1:
                transpose_layers_mag += [nn.ConvTranspose2d(
                    in_channels=decoder_mag[-1].calculate_output_chans(),
                    out_channels=decoder_mag[-1].calculate_output_chans(),
                    kernel_size=2,
                    stride=2
                    )]
                transpose_layers_phase += [nn.ConvTranspose2d(
                    in_channels=decoder_phase[-1].calculate_output_chans(),
                    out_channels=decoder_phase[-1].calculate_output_chans(),
                    kernel_size=2,
                    stride=2
                    )]
                previous_outchans = decoder_mag[-1].calculate_output_chans()
        
        self.decoder_mag = nn.ModuleList(decoder_mag)
        self.decoder_phase = nn.ModuleList(decoder_phase)
        self.transpose_layers_mag = nn.ModuleList(transpose_layers_mag)
        self.transpose_layers_phase = nn.ModuleList(transpose_layers_phase)
        
        self.decision_mag = nn.Conv2d(in_channels=decoder_mag[-1].calculate_output_chans(),
                                out_channels=self.output_chans,
                                kernel_size=3,
                                padding=1)
        self.decision_phase = nn.Conv2d(in_channels=decoder_phase[-1].calculate_output_chans(),
                                out_channels=self.output_chans,
                                kernel_size=3,
                                padding=1)
        self.decision_activation = nn.Tanh()

    def forward(self, x):
        stack = []
        for idx, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            #print(f'encode {idx+1} : {x.shape}')
            stack += [x]
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        stack = stack[::-1]
        m = self.decoder_mag[0](stack[0])
        p = self.decoder_phase[0](stack[0])
        #print(f'x : {x.shape}')
        for idx, (t_m_layer, t_p_layer) in \
                enumerate(zip(self.transpose_layers_mag, self.transpose_layers_phase)):
            ym = t_m_layer(m)
            yp = t_p_layer(p)
            m = torch.cat([stack[1 + idx], ym], dim=1)
            p = torch.cat([stack[1 + idx], yp], dim=1)
            m = self.decoder_mag[1 + idx](m)
            p = self.decoder_phase[1 + idx](p)
            #print(f'decoder {x.shape}')
        m = self.decision_mag(m)
        m = self.decision_activation(m)
        p = self.decision_phase(p)
        p = self.decision_activation(p)
        return m, p
# %%
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet_magphase_split(input_chans=24,
            output_chans=16,
            n_layers=[4,4,4,4],
            filters=[32, 64, 128, 256]).to(device)
torchsummary.summary(model, (24, 64, 128))
"""
# %%
class Discriminator(nn.Module):
    def __init__(self,
                input_chans=24,
                n_layers=[3, 4, 4, 3],
                filters=[4,4,4,4]):
        super().__init__()
        self.input_chans=input_chans
        self.n_layers=n_layers
        self.filters=filters

        #Define Encoder
        encoder = []
        decoder_inchans = []
        in_chans = self.input_chans
        for i, (n_filter, n_layer) in enumerate(zip(self.filters, self.n_layers)):
            encoder += [dense_block(in_chans=in_chans, out_chans=n_filter, n_layers=n_layer)]
            in_chans = encoder[-1].calculate_output_chans()
            decoder_inchans += [in_chans]

        self.encoder = nn.ModuleList(encoder)
        self.dense_1 = nn.Linear(in_chans, 512)
        self.dense_1_act = nn.ReLU()
        self.dense_2 = nn.Linear(512, 1)
    
    def forward(self, x):
        for idx, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Global average Pool
        x = F.avg_pool2d(x, kernel_size=(x.shape[-1], x.shape[-2]))
        
        x = x.view(x.shape[0], -1)
        x = self.dense_1(x)
        x = self.dense_1_act(x)
        x = self.dense_2(x)
        x = F.sigmoid(x)
        return x

#device = "cuda" if torch.cuda.is_available() else "cpu"
#model = Discriminator(input_chans=24,
#                        n_layers=[3,4,4,3],
#                        filters=[32,64,128,256]).to(device)
#torchsummary.summary(model, (24, 128, 128))

# %%
