#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os.path
import librosa
from scipy.io import wavfile
from data.base_dataset import BaseDataset
import h5py
import random
from random import randrange
import glob
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torchvision.transforms as transforms
import torch
from utils.lipreading_preprocess import *
import time
from utils.video_reader import VideoReader
#归一化函数/正则化图像函数
def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples
#视频预处理管道
def get_preprocessing_pipelines():
    # -- preprocess for the video stream  视频流预处理
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)                                #裁剪尺寸
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([                           #函数式编程函数compose
                                Normalize( 0.0,255.0 ),          #正则化图像函数
                                RandomCrop(crop_size),           #图片随机裁剪
                                HorizontalFlip(0.5),             #水平翻转图像
                                Normalize(mean, std) ])
    preprocessing['val'] = Compose([
                                Normalize( 0.0,255.0 ),
                                CenterCrop(crop_size),          #中心裁剪
                                Normalize(mean, std) ])
    preprocessing['test'] = preprocessing['val']
    return preprocessing
#下载图像帧，输入是图像文件路径
def load_frame(clip_path):
    video_reader = VideoReader(clip_path, 1)
    start_pts, time_base, total_num_frames = video_reader._compute_video_stats()                 #获取图像数据
    end_frame_index = total_num_frames - 1                                                       #最后一帧下标
    if end_frame_index < 0:
        clip, _ = video_reader.read(start_pts, 1)                                                #read函数读取文件数据
    else:
        clip, _ = video_reader.read(random.randint(0, end_frame_index) * time_base, 1)           #random.randint（）用于生成指定范围内的随机整数
    frame = Image.fromarray(np.uint8(clip[0].to_rgb().to_ndarray())).convert('RGB')              #RGB: 3x8位像素，真彩色
    # convert() 用以指定一种色彩模式；Image.fromarray(np.uint8())函数：array转换成image；to_ndarray() 转化为numpy数据；to_RGB方法返回一个由遮罩剪辑生成的非遮罩剪辑
    return frame

def get_mouthroi_audio_pair(mouthroi, audio, window, num_of_mouthroi_frames, audio_sampling_rate):
    audio_start = randrange(0, audio.shape[0] - window + 1)                                      #randrange() 方法返回指定递增基数集合中的一个随机数，基数默认值为1
    audio_sample = audio[audio_start:(audio_start+window)]
    frame_index_start = int(round(audio_start / audio_sampling_rate * 25))
    mouthroi = mouthroi[frame_index_start:(frame_index_start + num_of_mouthroi_frames), :, :]
    return mouthroi, audio_sample

def load_mouthroi(filename):
    try:
        #python保存数据方式（npy, pkl, h5, pt, npz）
        if filename.endswith('npz'):
            return np.load(filename)['data']
        elif filename.endswith('h5'):
            with h5py.File(filename, 'r') as hf:
                return hf["data"][:]
        else:
            return np.load(filename)
    except IOError:
        print( "Error when reading file: {}".format(filename) )
        sys.exit()
#生成语谱图，获得相应的幅值和相位
#音频特征提取——librosa工具包
def generate_spectrogram_magphase(audio, stft_frame, stft_hop, n_fft):
    # librosa.core.stft（）实现stft并绘制时频谱
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)                       #librosa.core.magphase（）获取相位和幅值
    spectro_mag = np.expand_dims(spectro_mag, axis=0)                                 #扩展维度
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag
#转化为复数
def generate_spectrogram_complex(audio, stft_frame, stft_hop, n_fft):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)                            #numpy.concatenate()函数能够一次完成多个数组的拼接
    return spectro_two_channel
#图像增强
def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)                     #Image.transpose() 转移图像(以90度为单位翻转或旋转)
    enhancer = ImageEnhance.Brightness(image)                              #可用于控制图像的亮度
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)                                   #用于调整图像的色彩平衡
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image
#音频增强
def augment_audio(audio):
    audio = audio * (random.random() * 0.2 + 0.9) # 0.9 - 1.1
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

class AudioVisualDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.audio_window = int(opt.audio_length * opt.audio_sampling_rate)                 #设置音频窗口大小
        #random.seed() 函数用于指定随机数生成时所用算法开始的整数值
        random.seed(opt.seed)
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[opt.mode]
        
        #load videos path from hdf5 file
        h5f_path = os.path.join(opt.data_path, opt.mode+'.h5') 
        h5f = h5py.File(h5f_path, 'r')
        self.videos_path = list(h5f['videos_path'][:])
        self.videos_path = [x.encode("utf-8").decode("utf-8") for x in self.videos_path]

        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.Resize(224), transforms.ToTensor()]
        if self.opt.normalization:
            vision_transform_list.append(normalize)
        self.vision_transform = transforms.Compose(vision_transform_list)

    def __getitem__(self, index):
        videos2Mix = random.sample(self.videos_path, 2) #get two videos
        #sample two clips for speaker A  扬声器A的两个剪辑样本
        videoA_clips = os.listdir(videos2Mix[0])
        clipPair_A = random.choices(videoA_clips, k=2) #randomly sample two clips  随机抽取两个剪辑
        #clip A1
        video_path_A1 = os.path.join(videos2Mix[0], clipPair_A[0])
        mouthroi_path_A1 = os.path.join(videos2Mix[0].replace('/mp4/', '/mouth_roi/'), clipPair_A[0].replace('.mp4', '.h5'))
        audio_path_A1 = os.path.join(videos2Mix[0].replace('/mp4/', '/audio/'), clipPair_A[0].replace('.mp4', '.wav'))
        #clip A2
        video_path_A2 = os.path.join(videos2Mix[0], clipPair_A[1])
        mouthroi_path_A2 = os.path.join(videos2Mix[0].replace('/mp4/', '/mouth_roi/'), clipPair_A[1].replace('.mp4', '.h5'))
        audio_path_A2 = os.path.join(videos2Mix[0].replace('/mp4/', '/audio/'), clipPair_A[1].replace('.mp4', '.wav'))
        #sample one clip for person B
        videoB_clips = os.listdir(videos2Mix[1])
        clipB = random.choice(videoB_clips) #randomly sample one clip
        video_path_B = os.path.join(videos2Mix[1], clipB)
        mouthroi_path_B = os.path.join(videos2Mix[1].replace('/mp4/', '/mouth_roi/'), clipB.replace('.mp4', '.h5'))
        audio_path_B = os.path.join(videos2Mix[1].replace('/mp4/', '/audio/'), clipB.replace('.mp4', '.wav'))

        #start_time = time.time()
        mouthroi_A1 = load_mouthroi(mouthroi_path_A1)
        mouthroi_A2 = load_mouthroi(mouthroi_path_A2)
        mouthroi_B = load_mouthroi(mouthroi_path_B)
        _, audio_A1 = wavfile.read(audio_path_A1)
        _, audio_A2 = wavfile.read(audio_path_A2)
        _, audio_B = wavfile.read(audio_path_B)
        audio_A1 = audio_A1 / 32768
        audio_A2 = audio_A2 / 32768
        audio_B = audio_B / 32768

        if not (len(audio_A1) > self.audio_window and len(audio_A2) > self.audio_window and len(audio_B) > self.audio_window):
            return self.__getitem__(index)
        
        mouthroi_A1, audio_A1 = get_mouthroi_audio_pair(mouthroi_A1, audio_A1, self.audio_window, self.opt.num_frames, self.opt.audio_sampling_rate)
        mouthroi_A2, audio_A2 = get_mouthroi_audio_pair(mouthroi_A2, audio_A2, self.audio_window, self.opt.num_frames, self.opt.audio_sampling_rate)
        mouthroi_B, audio_B = get_mouthroi_audio_pair(mouthroi_B, audio_B, self.audio_window, self.opt.num_frames, self.opt.audio_sampling_rate)
        
        frame_A_list = []
        frame_B_list = []
        for i in range(self.opt.number_of_identity_frames):
            frame_A = load_frame(video_path_A1)
            frame_B = load_frame(video_path_B)
            if self.opt.mode == 'train':
                frame_A = augment_image(frame_A)
                frame_B = augment_image(frame_B)
            frame_A = self.vision_transform(frame_A)
            frame_B = self.vision_transform(frame_B)
            frame_A_list.append(frame_A)
            frame_B_list.append(frame_B)
        frames_A = torch.stack(frame_A_list).squeeze()
        frames_B = torch.stack(frame_B_list).squeeze() 

        if not (mouthroi_A1.shape[0] == self.opt.num_frames and mouthroi_A2.shape[0] == self.opt.num_frames and mouthroi_B.shape[0] == self.opt.num_frames):
            return self.__getitem__(index)

        #transform mouthrois and audios  转换唇读和音频
        mouthroi_A1 = self.lipreading_preprocessing_func(mouthroi_A1)
        mouthroi_A2 = self.lipreading_preprocessing_func(mouthroi_A2)
        mouthroi_B = self.lipreading_preprocessing_func(mouthroi_B)
        
        #transform audio
        if(self.opt.audio_augmentation and self.opt.mode == 'train'):
            audio_A1 = augment_audio(audio_A1)
            audio_A2 = augment_audio(audio_A2)
            audio_B = augment_audio(audio_B)
        if self.opt.audio_normalization:
            audio_A1 = normalize(audio_A1)
            audio_A2 = normalize(audio_A2)
            audio_B = normalize(audio_B)
                
        #get audio spectrogram  获取音频频谱图
        audio_mix1 = (audio_A1 + audio_B) / 2
        audio_mix2 = (audio_A2 + audio_B) / 2
        #generate_spectrogram_complex函数：把频谱图转换成复数
        audio_spec_A1 = generate_spectrogram_complex(audio_A1, self.opt.window_size, self.opt.hop_size, self.opt.n_fft)
        audio_spec_A2 = generate_spectrogram_complex(audio_A2, self.opt.window_size, self.opt.hop_size, self.opt.n_fft)
        audio_spec_B = generate_spectrogram_complex(audio_B, self.opt.window_size, self.opt.hop_size, self.opt.n_fft)
        audio_spec_mix1 = generate_spectrogram_complex(audio_mix1, self.opt.window_size, self.opt.hop_size, self.opt.n_fft)
        audio_spec_mix2 = generate_spectrogram_complex(audio_mix2, self.opt.window_size, self.opt.hop_size, self.opt.n_fft)
        
        data = {}
        #torch.FloatTensor函数可以将变量转化为浮点型32位
        data['mouthroi_A1'] = torch.FloatTensor(mouthroi_A1).unsqueeze(0)
        data['mouthroi_A2'] = torch.FloatTensor(mouthroi_A2).unsqueeze(0)
        data['mouthroi_B'] = torch.FloatTensor(mouthroi_B).unsqueeze(0)
        data['frame_A'] = frames_A
        data['frame_B'] = frames_B
        data['audio_spec_A1'] = torch.FloatTensor(audio_spec_A1)
        data['audio_spec_A2'] = torch.FloatTensor(audio_spec_A2)
        data['audio_spec_B'] = torch.FloatTensor(audio_spec_B)
        data['audio_spec_mix1'] = torch.FloatTensor(audio_spec_mix1)
        data['audio_spec_mix2'] = torch.FloatTensor(audio_spec_mix2)
        return data

    def __len__(self):
        if self.opt.mode == 'train':
            return self.opt.batchSize * self.opt.num_batch
        elif self.opt.mode == 'val':
            return self.opt.batchSize * self.opt.validation_batches

    def name(self):
        return 'AudioVisualDataset'        #返回数据集
