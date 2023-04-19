#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data
from data.base_data_loader import BaseDataLoader
from utils.utils import object_collate
#创建数据集
def CreateDataset(opt):
    dataset = None
    if opt.model == 'audioVisual':
        from data.audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))                 #打印数据集名字为‘AudioVisualDatase’
    dataset.initialize(opt)                                              #初始化数据集参数
    return dataset                                                       #返回创建好的数据集
#加载数据集
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)                        #初始化参数
        self.dataset = CreateDataset(opt)                           #创建数据集
        if opt.mode == "train":
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=False,
                #num_workers=int(opt.nThreads),
                num_workers=8,
                collate_fn=object_collate)                            #加载创建好的数据集，并自定义相关参数
        elif opt.mode == 'val':
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=False,
                num_workers=2,
                collate_fn=object_collate)          

    def load_data(self):                             #返回数据集
        return self

    def __len__(self):                               #返回加载的数据集长度
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):   #enumerate表示遍历一个集合对象
            yield data
