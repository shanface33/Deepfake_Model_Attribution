# encoding:utf-8
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import data
from collections import OrderedDict
import os
import torch.backends.cudnn as cudnn
import math
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from data.dataset import MyDataset
from models import DMA_STA
from config import opt
import numpy as np
import scipy.io as sio
from PIL import Image
import random
import argparse
import sys

acc_list = [0.0]
loss_list = [0.0]

criterion = nn.CrossEntropyLoss()

def train(epoch, lr):
    epoch_start = time.time()

    features_lr = lr * 0.1
    if features_lr <= 0.0001:
        features_lr = 0.0001
    optimizer = optim.SGD(
       [
           {'params': model.parameters(), 'lr': features_lr},\
       ],
       lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    model.train()
    start = time.time()
    running_loss = 0.0

    train_bs = args.train_bs
    train_len = len(trainset)

    for batch_idx, data0 in enumerate(trainloader, 1):
        data, target = data0
        data = Variable(data)
        target = Variable(target)
        data, target = data.cuda(), target.cuda()  
        output = model(data)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        running_loss += loss.data.item()
        optimizer.step()

    loss_tmp = running_loss / (train_len / args.train_bs)  # div the n of batch
    print('Epoch:{}[{}/{} ]\t Loss:{:.3f}\t LR:{}'.format(   
                 epoch, batch_idx * len(data), train_len, loss_tmp, lr))
    epoch_end = time.time()
    tmp = (epoch_end - epoch_start) / 60
    print('Epoch:{}, train time:{:.4f} min'.format(epoch, tmp))


def test(epoch):
    model.eval()
    test_loss = 0
    fin_preds = []
    fin_trues = []
    prediction_list = []
    start = time.time()
    test_bs = args.test_bs
    test_len = len(testset)
    for batch_idx, data0 in enumerate(testloader,1):
        data, target = data0
        data = Variable(data)
        target = Variable(target)
        data, target = data.cuda(), target.cuda()

        with torch.no_grad():  
            output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]  
        
        fin_preds.extend(pred.detach().cpu().numpy())
        fin_trues.extend(target.detach().cpu().numpy())

    #print(prediction_list)
    test_loss = test_loss / (test_len / args.test_bs)
    sklearn_accuf = accuracy_score(fin_trues, fin_preds)
    conf_matrixf = get_confusion_matrix(5, fin_trues, fin_preds)

    interval = time.time() - start
    print('\nTest average loss: {:.4f}\t Accuracy: {:.2f}'.format(test_loss, sklearn_accuf*100))
    print('\n', conf_matrixf)

    model_name = 'tmp/' + str(epoch) + '_' + str(sklearn_accuf) + '.pth'
    acc_max = max(acc_list)
    if sklearn_accuf > acc_max and sklearn_accuf > 20:
        torch.save(model.state_dict(), model_name)
        print('I have saved the model')

    acc_list.append(sklearn_accuf)
    acc_max = max(acc_list)
    print('max acc:{} in epoch {}'.format(acc_max, acc_list.index(acc_max)))
  
def get_confusion_matrix(n_lab, trues, preds):
    if n_lab == 5:
        labels = [0,1,2,3,4]
    elif n_lab == 6:
        labels = [0,1,2,3,4,5]
    conf_matrix = confusion_matrix(trues, preds) 
    return conf_matrix
    
def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-res_plus', '--res_plus', type=int, default=opt.res_plus)
    parser.add_argument('-res', '--res', type=int, default=opt.res)
    parser.add_argument('-lr', '--lr', type=float, default=opt.lr)
    parser.add_argument('-lr_scale', '--lr_scale', type=float, default=opt.lr_scale)
    parser.add_argument('-train_bs', '--train_bs', type=int, default=opt.train_bs)
    parser.add_argument('-device', '--gpu_device', default=opt.gpu_device)
    parser.add_argument('-save_low_bound', '--save_low_bound', type=float, default=opt.save_low_bound)
    parser.add_argument('-weight_decay', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('-test_bs', '--test_bs', type=int, default=opt.test_bs)
    parser.add_argument('-test_epoch', '--test_epoch', type=int, default=opt.test_epoch)
    parser.add_argument('-pretrained', '--pretrained', default=opt.pretrained)
    parser.add_argument('-pre_model_path', '--pre_path', default=opt.pre_path)
    parser.add_argument('-use_gpu', '--use_gpu', default=opt.use_gpu)
    parser.add_argument('-max_epoches', '--max_epoches', type=int, default=opt.max_epoches)
    args = parser.parse_args()
    return args


def main(argv):
    global args
    global model
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    print('model:', 'DMA_STA')
    model = DMA_STA()
    model.cuda()

    if True:
        global trainset
        train_transforms = transforms.Compose([
            transforms.Resize(args.res_plus),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(args.res),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.643, 0.466, 0.387), std=(0.259, 0.220, 0.203))
        ])
        trainset = MyDataset('../data/Frame_10_cfr10_train', 10, train_transforms) # 10-frame number
        print(len(trainset))
        train_bs = args.train_bs
        global trainloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

        train_transforms2 = transforms.Compose([
            transforms.Resize(args.res_plus),
            #transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(args.res),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.643, 0.466, 0.387), std=(0.259, 0.220, 0.203))
        ])
        global testset
        testset = MyDataset('../data/Frame_10_cfr10_test', 10, train_transforms2)
        print(len(testset))
        test_bs = args.test_bs
        global testloader
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    lr = args.lr
    mode = True  # 1 : train 0: test
    if mode:
        for epoch in range(1, args.max_epoches):
            if epoch in opt.lr_freq_list:
                lr = lr * args.lr_scale
                lr = max(lr, 0.0001)
            train(epoch, lr)
            if epoch % args.test_epoch == 0:
                test(epoch)
    else:
        test()


if __name__ == '__main__':
    main(sys.argv[1:])




