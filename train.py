# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from models.MF import MF
from data.dataset import H5Dataset
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.loss import Fusionloss, cc, SSIMLoss
import kornia
from skimage.metrics import structural_similarity as ssim
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()

num_epochs = 1  # total epoch

lr = 1e-4
weight_decay = 0
batch_size = 4

GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1.  # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.  # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MF().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)

# ssim_loss = SSIMLoss(data_range=1.0)

# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_imgsize_64_stride_32.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

        model.train()

        model.zero_grad()

        optimizer.zero_grad()

        data_Fuse = model(data_VIS, data_IR)

        fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

        # loss_ssim = ssim_loss(data_Fuse, data_IR, data_VIS)

        # loss = fusionloss + loss_ssim
        loss = fusionloss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

        scheduler.step()

        if optimizer.param_groups[0]['lr'] <= 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6

if True:
    checkpoint = {
        "epoch": epoch,
        'model_state_dict': model.state_dict(),
        # "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join("models/MSRS-unet-6-" + str(epoch+1) + '.pth'))