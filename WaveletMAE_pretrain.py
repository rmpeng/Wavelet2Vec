
import torch
import argparse
from train_models.Wavelet2Vec import Wavelet_Transformer
from utils.utils import adjust_learning_rate

import pickle
import time
import pandas as pd
import torch, shutil
import numpy as np
import my_config

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from my_config import Config
import os
from base_models.base_wavelet_decompose import nd_Wavelet_reconstruct

import sklearn.metrics as sm
import warnings
from utils.pytorchtools import EarlyStopping
from torch.utils.data.sampler import WeightedRandomSampler
import os.path
import joblib
from lib import train_epoch, val,val_former, print_time_cost, \
    plotCurve, old_reshape_input, vote, plot_confusion_matrix, ensemble_test
import my_config
GPU_ID = my_config.Config.GPU_id
device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else "cpu")

torch.manual_seed(1042)
torch.cuda.manual_seed(1042)

import matplotlib.pyplot as plt

def pre_train_waveletmae(home_dir,patch_size,fs, train_dataloader, epochs =200, reshape_flag = False):
    model = Wavelet_Transformer(patch_size=patch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    mask_ratio = 0.1
    interval = 50

    for epoch in range(1, epochs+1):
        total_loss = []
        total_acc = []
        model.train()
        for inputs, target, sweight, class_weight in train_dataloader:

            inputs = inputs  # Batchsize * channel * 1 * points
            if reshape_flag is True:
                inputs = old_reshape_input(inputs)  # Batchsize * 1 *Channels * timepoin
            target = target.to(device)
            target2 = torch.argmax(target, dim=1)
            # zero the parameter gradients
            optimizer.zero_grad()

            Data,delta,theta,alpha,beta,gamma,upper = nd_Wavelet_reconstruct(inputs, fs=fs,wavelet='db4',max_level=7)
            # Data.to(device)
            # delta.to(device)
            # theta.to(device)
            # alpha.to(device)
            # beta.to(device)
            # gamma.to(device)
            # upper.to(device)
            loss, pred, mask = model(Data, delta,theta,alpha, beta,gamma,upper, mask_ratio)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        adjust_learning_rate(optimizer, epoch, Config.lr, epochs)
        print("Training->Epoch:{:0>2d}, Loss:{:.3f}".format(epoch, torch.tensor(total_loss).mean()))
        if epoch % interval == 0:
            chkpoint = {'model': model.state_dict()}
            torch.save(chkpoint, os.path.join(home_dir, str('epoch'+str(epoch)+'_chkpoint.pt')))

    return home_dir






