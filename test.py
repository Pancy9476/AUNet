import os
import numpy as np
import torch
import torch.nn as nn
from data.img_read_save import img_save,image_read_cv2
import warnings
import logging
from models.MF import MF
import cv2
from models.MF import MF
from data.dataset import H5Dataset
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.loss import Fusionloss, cc
import kornia
from skimage.metrics import structural_similarity as ssim
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# pet-mri = 299  spect-mri = 100 msrs = 60
ckpt_path= r"models/MSRS-60.pth"
# ckpt_path= r"models/MSRS-unet-6-1.pth"
# ckpt_path= r"models/CT-100.pth"
# ckpt_path= r"models/SPECT-MRI-100.pth"

for dataset_name in ["MSRS"]:
    print("The test result of " + dataset_name + " :")
    # test_folder = os.path.join('datastes', dataset_name, 'test')
    test_folder = os.path.join('datastes', dataset_name)
    test_out_folder = os.path.join('test_result', dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MF().to(device)
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    total_params = (sum(p.numel() for p in model.parameters())) / 1e6
    # print(f"Total params: {total_params :.6f}M")
    # a = sum(p.numel() for p in model.downs.parameters())/1000000
    # b = sum(p.numel() for p in model.mid.parameters())/1000000
    # c = sum(p.numel() for p in model.ups.parameters())/1000000
    # d = sum(p.numel() for p in model.decoder.parameters())/1000000
    model.eval()

    st = time.time()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "ir")):
            # print(f"Image: {img_name}.")
            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='YCrCb'))[0][np.newaxis, np.newaxis, ...] / 255.0
            # ycrcb, uint8
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))
            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            data_Fuse = model(data_VIS, data_IR)

            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))

            fi = np.squeeze((data_Fuse * 255.0).cpu().numpy())

            # float32 to uint8
            fi = fi.astype(np.uint8)
            # concatnate
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)
            print(f"Image: {img_name} is done.")

    et = time.time()
    print((et - st) / len(os.listdir(os.path.join(test_folder, "ir"))))