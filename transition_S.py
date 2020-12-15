import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
K = 8
M = 8
Ls = 128

#S_mat_path = "/data/datasets/hyundai/20200619_DNN_ANC_sample_secondary_path/S_128_8_8.mat"
S_mat_path = "/skuser/data/S_128_8_8.mat"
S_mat = loadmat(S_mat_path, mat_dtype=True)

S_data = S_mat['S'] # (Ls, K, M)

def conv_s(y, s_filter):
    from tqdm import tqdm
    # defined as a function
    ## New Conv S => Validated
    Ls, K, M = s_filter.shape
    Tn = y.shape[0]
    y_buffer = torch.zeros((Ls, K))
    y_p = torch.zeros(y.size())
    y_pp = torch.zeros(y.size())
    #e = torch.zeros(y.size())

    for n in tqdm(range(Tn)):
        for k in range(K):
            for m in range(M):
                y_p[n,m] += torch.dot(y_buffer[:, k].transpose(1,0), s_filter[:, k, m])
                #y_pp[n,m] += y_buffer[:, k] * s_filter[:, k, m]

        #e[n, :] = d[n, :] - y_p[n, :]
        y_buffer[1:, :] = y_buffer[:-1, :].clone()
        y_buffer[0, :] = y[n , :]
    return y_p#, y_pp

class S_conv(nn.Module):
    def __init__(self, s_filter=S_data, device="cuda"):
        super().__init__()
        self.device = device
        self.S_data = torch.tensor(s_filter, dtype=torch.float64).to(device)
        self.K = 8
        self.M = 8
        self.Ls = 128

    def padding(self, signal):
        _pad = torch.zeros((signal.size(0), self.Ls, signal.size(2)),
                                device=self.device, dtype=torch.float64)
        return torch.cat([_pad, signal],1)
    
    def forward(self, signal):
        # S_data(Ls, K, M)
        device = self.device
        s_data = torch.tensor(self.S_data.transpose(0,1).cpu().numpy()[:,::-1,:].copy(),
                                device=device, dtype=torch.float64)
        signal = self.padding(signal)
        if signal.size(1) != self.K:
            signal = signal.transpose(1,2)
        out = F.conv1d(signal, s_data.permute([2,0,1]))

        return out.transpose(1,2)[:,:-1,:]