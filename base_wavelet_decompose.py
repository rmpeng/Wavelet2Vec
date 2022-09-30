import matplotlib.pyplot as plt
import pywt
import mne
import numpy as np
import my_config
GPU_ID = my_config.Config.GPU_id
import torch
device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else "cpu")

import torch

iter_freqs = [
    {'name': 'Delta', 'fmin': 0, 'fmax': 4},
    {'name': 'Theta', 'fmin': 4, 'fmax': 8},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 16},
    {'name': 'Beta', 'fmin': 16, 'fmax': 32},
    {'name': 'gamma', 'fmin': 32, 'fmax': 64},
    {'name': 'upper', 'fmin': 64, 'fmax': 128},
]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
mne.set_log_level(False)

def TimeFrequencyWP(data, fs, wavelet, maxlevel = 8):
    # 小波包变换这里的采样频率为250，如果maxlevel太小部分波段分析不到
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs/(2**maxlevel)
    sum = None
    #######################根据实际情况计算频谱对应关系，这里要注意系数的顺序
    # 绘图显示
    fig, axes = plt.subplots(len(iter_freqs)+2, 1, figsize=(10, 7), sharex=True, sharey=False)
    # 绘制原始数据
    axes[0].plot(data, color='black')
    axes[0].set_title('signal')
    sum = 0
    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin']<=bandMin and iter_freqs[iter]['fmax']>= bandMax):
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = wp[freqTree[i]].data
        # 绘制对应频率的数据
        sum = sum + new_wp.reconstruct(update=True) if sum is not None else new_wp.reconstruct(update=True)
        axes[iter+1].plot(new_wp.reconstruct(update=True)[:512], color='black')
        # 设置图名
        #axes[iter+1].set_title(iter_freqs[iter]['name'])
    axes[len(iter_freqs) + 1].plot(sum[:512], color='black')
    # axes[len(iter_freqs) + 1].set_title('reconstruct')
    plt.show()
    print(sum[:512] - data)

def sig_wavelet_decomposer(signal, fs, wavelet, signal_length=512, max_level = 8):
    new_signals = {'Delta':[],
                   'Theta':[],
                   'Alpha': [],
                   'Beta': [],
                   'gamma': [],
                   'upper': []
                   }

    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
    freqTree = [node.path for node in wp.get_level(max_level, 'freq')]
    freqBand = fs/(2**max_level)
    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin']<=bandMin and iter_freqs[iter]['fmax']>= bandMax):
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = wp[freqTree[i]].data
        # 绘制对应频率的数据
        wavename = iter_freqs[iter]['name']
        new_signals[wavename] = new_wp.reconstruct(update=True)[:signal_length]

    return new_signals

def Wavelet_reconstruct(Data, fs, wavelet, max_level = 8):
    Data = Data.numpy()
    batch_size = Data.shape[0]
    channel_num = Data.shape[1]
    signal_length = Data.shape[3]

    tmp_delta = np.zeros([batch_size, channel_num, signal_length])
    tmp_theta = np.zeros([batch_size, channel_num, signal_length])
    tmp_alpha = np.zeros([batch_size, channel_num, signal_length])
    tmp_beta = np.zeros([batch_size, channel_num, signal_length])
    tmp_gamma = np.zeros([batch_size, channel_num, signal_length])
    tmp_upper = np.zeros([batch_size, channel_num, signal_length])


    for bch in range(batch_size):
        for ch in range(channel_num):
            signal = np.squeeze(Data[bch, ch, :, :])
            reconstruct_sig = sig_wavelet_decomposer(signal, fs, wavelet, signal_length=signal_length, max_level = max_level)
            tmp_delta[bch, ch, :] = reconstruct_sig['Delta']
            tmp_theta[bch, ch, :] = reconstruct_sig['Theta']
            tmp_alpha[bch, ch, :] = reconstruct_sig['Alpha']
            tmp_beta[bch, ch, :] = reconstruct_sig['Beta']
            tmp_gamma[bch, ch, :] = reconstruct_sig['gamma']
            tmp_upper[bch, ch, :] = reconstruct_sig['upper']

    delta = np.expand_dims(tmp_delta, axis=2)
    theta = np.expand_dims(tmp_theta, axis=2)
    alpha = np.expand_dims(tmp_alpha, axis=2)
    beta = np.expand_dims(tmp_beta, axis=2)
    gamma = np.expand_dims(tmp_gamma, axis=2)
    upper = np.expand_dims(tmp_upper, axis=2)

    Data = torch.Tensor(Data).to(device)
    delta = torch.Tensor(delta).to(device)
    theta = torch.Tensor(theta).to(device)
    alpha = torch.Tensor(alpha).to(device)
    beta = torch.Tensor(beta).to(device)
    gamma = torch.Tensor(gamma).to(device)
    upper = torch.Tensor(upper).to(device)

    return Data,delta,theta,alpha,beta,gamma,upper



def nd_Wavelet_reconstruct(Data, fs, wavelet, max_level = 8):
    Data = Data.numpy()
    batch_size = Data.shape[0]
    channel_num = Data.shape[1]
    signal_length = Data.shape[3]

    new_signals = {'Delta': [],
                   'Theta': [],
                   'Alpha': [],
                   'Beta': [],
                   'gamma': [],
                   'upper': []
                   }

    wp = pywt.WaveletPacketND(data=Data, wavelet=wavelet, mode='symmetric', maxlevel=max_level, axes=3)
    freqTree = [node.path for node in wp.get_level(max_level, 'freq')]
    freqBand = fs / (2 ** max_level)
    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacketND(data=None, wavelet=wavelet, mode='symmetric', maxlevel=max_level, axes=3)
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = wp[freqTree[i]].data
        # 绘制对应频率的数据
        wavename = iter_freqs[iter]['name']
        new_signals[wavename] = new_wp.reconstruct(update=True)[:,:,:,:signal_length]

    Data = torch.Tensor(Data).to(device)
    delta = torch.Tensor(new_signals['Delta']).to(device)
    theta = torch.Tensor(new_signals['Theta']).to(device)
    alpha = torch.Tensor(new_signals['Alpha']).to(device)
    beta = torch.Tensor(new_signals['Beta']).to(device)
    gamma = torch.Tensor(new_signals['gamma']).to(device)
    upper = torch.Tensor(new_signals['upper']).to(device)

    return Data,delta,theta,alpha,beta,gamma,upper








if __name__ == '__main__':
    dataCom = np.random.random(512)
    TimeFrequencyWP(dataCom,128,wavelet='db4', maxlevel=7)

