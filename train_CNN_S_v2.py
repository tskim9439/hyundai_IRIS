#%%
import os
import sys
sys.path.append('lib')
sys.path.append('models')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import easydict

import WaveNet_no_act as WaveNet
import models_without_act
import metric
import transition_S
# %%
config = easydict.EasyDict({
    # Data Path
    "base_path" : "/skuser/data",
    "snd_pickle" : "/skuser/data/stationary_accel_data.pickle",
    "acc_pickle" : "/skuser/data/stationary_sound_data.pickle",
    "snd_train_npy" : "train_snd.npy",
    "snd_test_npy" : "test_snd.npy",
    "acc_train_npy" : "train_acc.npy",
    "acc_test_npy" : "test_acc.npy",
    "s_filter_path" : "/skuser/data/S_128_8_8.mat",
    # Settings
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
    "save_pt" : "ANC_CNN__S_filter_custom_loss_v1.pth",
    # Data Preprocess
    "sampling_rate" : 8192,
    "sample_length" : 8128,
    "n_fft" : 256,
    "win_length" : 256,
    "hop_length" : 64,
    "epsilon" : 1e-6,
    "acc_max" : 0.5,
    "acc_min" : -0.5,
    "snd_max" : 7.5,
    "snd_min" : -7.5,
    # Train
    "batch_size" : 32,
    "epoch" : 500,
    "lr" : 1e-3,
    # Model
    "layer_size" : 8,
    "stack_size" : 1,
    "n_samples" : 3048,
    "step" : 4
})
#%%
# LOAD DATA
print("Data Loading...")
train_acc = np.load(config.acc_train_npy)
train_snd = np.load(config.snd_train_npy)

test_acc = np.load(config.acc_test_npy)
test_snd = np.load(config.snd_test_npy)

print("Data Loaded")
print(f'train ACC {train_acc.shape} SND {train_snd.shape}')
print(f'test  ACC {test_acc.shape}  SND {test_snd.shape}')

#%%
### Models ###
model = models_without_act.Pyramid(layer_filters=[64, 128, 128, 256],
                kernel_sizes=[256, 128, 128, 64, 32]).to(config.device)
receptive_field = model.calc_receptive_field()
# Input : (time, channels)
# output : (-1, 1, time)
#torchsummary.summary(model, (12, 2548))
# %%
torch.autograd.set_detect_anomaly(True)
class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data, config=config,
                receptive_field=255, step=4,#483,
                acc_normalizer=None, snd_normalizer=None):
        self.x_data = x_data
        self.y_data = y_data
        self.config = config
        self.receptive_field = receptive_field
        self.step = step
        self.acc_normalizer = acc_normalizer
        self.snd_normalizer = snd_normalizer

        print(f'x : {self.x_data.shape} y : {self.y_data.shape}')
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx].T) # (time, channels)
        y = torch.tensor(self.y_data[idx].T) # (time, channels)

        possible_max_idx = len(x) - self.config.n_samples - self.config.step
        rand_idx = np.random.randint(possible_max_idx)
        x = x[rand_idx : rand_idx + self.config.n_samples]
        y = y[rand_idx + self.config.step + self.receptive_field : rand_idx + self.config.step + self.config.n_samples]

        if self.acc_normalizer is not None:
            x = self.acc_normalizer(x)
        if self.snd_normalizer is not None:
            y = self.snd_normalizer(y)
        #print(x.shape, y.shape)
        #y = y.contiguous().transpose(1,0) # (channel, time)
        x = x.contiguous().transpose(1,0) # (channel, time)
        return x, y

train_dataset = Custom_Dataset(x_data=train_acc, y_data=train_snd, receptive_field=receptive_field, config=config,
                            acc_normalizer=None, snd_normalizer=None)

test_dataset = Custom_Dataset(x_data=test_acc, y_data=test_snd, receptive_field=receptive_field, config=config,
                            acc_normalizer=None, snd_normalizer=None)       

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                            shuffle=True,
                                            batch_size=config.batch_size)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1)

#%%
class Custom_Loss(nn.Module):
    def __init__(self, w=0.2):
        super().__init__()
        self.base_loss_fn = nn.SmoothL1Loss()
        self.w = w
    def forward(self, pred, gt):
        base_loss = self.base_loss_fn(pred, gt)
        amplitude_loss = torch.mean(torch.abs(pred) - torch.abs(gt))
        return base_loss + (amplitude_loss * self.w)

### Train ###
loss_fn = Custom_Loss() #nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr/2, betas=(0.5, 0.999))
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=3)

if os.path.isfile(config.save_pt):
    model.load_state_dict(torch.load(config.save_pt))
    print("Pretrained Model Loaded")

#%%
device = config.device
best_dba = -9999999.0
SP = transition_S.S_conv().to(device)

for epoch in range(config.epoch):
    model.train()
    with tqdm.tqdm(train_dataloader, ncols=100, desc='Epoch [%03d/%03d]'%(epoch+1, config.epoch)) as _tqdm:
        # Train
        train_losses = []
        for idx, (x, y) in enumerate(_tqdm):
            optimizer.zero_grad()
            x = x.double().to(device)
            y = y.double().to(device)
            #print(x.shape, y.shape)

            pred = model(x)
            pred = pred.contiguous().transpose(2,1)
            pred = SP(pred)

            #print(f'pred {pred.shape} gt {y.shape}')
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_losses += [loss.item()]
            _tqdm.set_postfix(loss=f'{np.mean(np.array(train_losses)):.5f}')
            if idx == 130:
                break
    
    model.eval()
    dBA = []
    val_losses = []
    with tqdm.tqdm(test_dataloader, ncols=100, desc='Validation') as _tqdm:
        for idx, (x, y) in enumerate(_tqdm):
            x = x.double().to(device)
            y = y.double().to(device)

            pred = model(x)
            pred = pred.contiguous().transpose(2,1)
            pred = SP(pred)

            loss = loss_fn(pred, y)

            val_losses += [loss.item()]

            pred_wav = pred.detach().cpu().numpy()[0]
            gt_wav   = y.detach().cpu().numpy()[0]

            if idx % 10 == 0:
                db, _ = metric.dBA_metric(y=pred_wav, gt=gt_wav, plot=True)
                print(f'dBA : {db}')
                plt.plot(pred_wav[100:-100, 0], color='blue')
                plt.plot(gt_wav[100:-100, 0], color='red')
                plt.show()
            else:
                db, _ = metric.dBA_metric(y=pred_wav, gt=gt_wav, plot=False)
            dBA += [db]
        
        dBA = np.mean(np.array(dBA))
        val_loss = np.mean(np.array(val_losses))
        print(f'EPOCH [{epoch+1}] best dBA {best_dba:.4f} val loss {val_loss:.4f} val dBA {dBA:.4f}')
        if dBA > best_dba:
            print(f'Best dBA has been improved from {best_dba:.4f} to {dBA:.4f}')
            best_dba = dBA
            torch.save(model.state_dict(), config.save_pt)
            print(f'Model saved {config.save_pt}')
        else:
            print(f'Did Not improved, best dBA {best_dba:.4f} current dBA {dBA:.4f}')
        lr_scheduler.step(int(val_loss))
        print(f"lr {lr_scheduler.optimizer.state_dict()['param_groups'][0]['lr']:.5f}")
# %%
