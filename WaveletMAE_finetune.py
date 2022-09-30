
import torch
import argparse
from train_models.wavelet_TiT_weighted import WaveletTimeTransformer
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
import sklearn.metrics as sm
import warnings
from utils.pytorchtools import EarlyStopping
from torch.utils.data.sampler import WeightedRandomSampler
import os.path
import joblib
from lib import train_epoch_wavelet, val_wavelet,val_former, print_time_cost, \
    plotCurve, old_reshape_input, vote, plot_confusion_matrix, ensemble_test
import my_config
GPU_ID = my_config.Config.GPU_id
device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else "cpu")

torch.manual_seed(1042)
torch.cuda.manual_seed(1042)

import matplotlib.pyplot as plt

def tune_train_waveletmae(pretrain_dir,patch_size, fs,tune_dir, train_dataloader,val_dataloader, reshape_flag = False):

    model = WaveletTimeTransformer(patch_size=patch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(pretrain_dir, 'epoch100_chkpoint.pt'))['model']

    del_list = ['pos_embed_delta', 'pos_embed_theta','pos_embed_alpha',
                'pos_embed_beta','pos_embed_gamma','pos_embed_upper',
                'mask_token_delta', 'mask_token_theta','mask_token_alpha',
                'mask_token_beta','mask_token_gamma','mask_token_upper',
                'decoder_embed_delta', 'decoder_embed_theta','decoder_embed_alpha',
                'decoder_embed_beta','decoder_embed_gamma','decoder_embed_upper',
                'decoder_blocks_delta','decoder_blocks_theta','decoder_blocks_alpha',
                'decoder_blocks_beta','decoder_blocks_gamma','decoder_blocks_upper',
                'decoder_pred_delta','decoder_pred_theta','decoder_pred_alpha',
                'decoder_pred_beta','decoder_pred_gamma','decoder_pred_upper',
                'patch_embed_delta','patch_embed_theta','patch_embed_alpha',
                'patch_embed_beta','patch_embed_gamma','patch_embed_upper'
                ]
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # train
    best_f1 = -1
    lr = Config.lr
    start_epoch = 1
    stage = 1

    early_stopping_t = EarlyStopping(patience=Config.patience, verbose=True,
                                     path=os.path.join(tune_dir, 'checkpoint.pt'))

    epc = 0
    train_ls, val_ls = [], []
    best_epoch = Config.max_epoch
    for epoch in range(start_epoch, Config.max_epoch + 1):
        epc += 1
        since = time.time()
        train_loss, train_f1 = train_epoch_wavelet(model,fs, optimizer, criterion,
                                           train_dataloader, show_interval=10,
                                           batch_size=Config.batch_size, device=device, reshape_flag=reshape_flag)
        val_output, val_loss, val_f1, val_bca = val_wavelet(model, fs, criterion, val_dataloader, device=device,
                                                    reshape_flag=reshape_flag)
        train_ls.append(train_loss)
        val_ls.append(val_loss)

        print('#epoch:%02d stage:%d train_loss:%.3e val_loss:%0.3e val_f1:%.3f val_bca:%.3f time:%s\n'
              % (epoch, stage, train_loss, val_loss, val_f1, val_bca, print_time_cost(since)))

        early_stopping_t(val_loss, model)

        if early_stopping_t.early_stop:
            print("Early stopping")
            best_epoch = epoch
            print('Best epoch is: ', best_epoch)
            break

    print("final epoch: train loss %f, val loss %f" % (train_ls[-1], val_ls[-1]))
    return  tune_dir




