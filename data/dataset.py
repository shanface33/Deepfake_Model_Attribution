#coding:utf-8
import os
import torch
import numpy as np
from torchvision import transforms,utils
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class MyDataset(Dataset):

    def __init__(self, root_dir, clip_len, transforms_=None, test_sample_num=1, stride=1):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        self.data = []
        self.stride = stride

        self.class2idx = {'df1': 0, 'df2': 1, 'df3': 2, 'df4': 3, 'df5': 4}  
        self.class_count = [0] * 5  
        self.fake_count = 0

        for base, subdirs, files in os.walk(self.root_dir):
            if len(files) < self.stride * self.clip_len:  # 
                continue
            data = {}
            video = []
            files.sort()
            for i, f in enumerate(files):
                if f.endswith('.png'):
                    data_dict = {}
                    data_dict['frame'] = os.path.join(base, f)
                    data_dict['index'] = i
                    video.append(data_dict)
            data['video'] = video
            data['class'] = self.class2idx[base.split('/')[-2]]
            self.class_count[data['class']] += 1
            data['label'] = 0 if 'df1' in base else 1 if 'df2' in base else 2 if 'df3' in base else 3 if 'df4' in base else 4
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        video = self.data[idx]['video']
        label = self.data[idx]['label']
        sub_class = self.data[idx]['class']
        length = len(video)

        clip_start = random.randint(0, length - (self.clip_len * self.stride))
        clip = video[clip_start: clip_start + (self.clip_len * self.stride): self.stride]

        if self.transforms_:
            trans_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            seed = random.random()
            for frame in clip:
                random.seed(seed)
                frame = Image.open(frame['frame'])
                frame = self.transforms_(frame)  # tensor [C x H x W]
                
                trans_clip.append(frame)
            # (T x C X H x W) to (C X T x H x W)
            clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
        else:
            clip = torch.tensor(clip)

        return clip, torch.tensor(int(label))

