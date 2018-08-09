import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import Config

import os

root = os.getcwd()

TrainPath = Config.TrainPath
ValPath = Config.ValPath
TestPath = Config.TestPath

useDistributed = Config.useDistributed

BatchSize = Config.BatchSize
Workers = Config.Workers

def DataLoader(mode='Train'):
    if mode=='Train':
        datapath = TrainPath
    elif mode=='Val':
        datapath = ValPath
    elif mode=='Test':
        datapath = TestPath
    else:
        raise ValueError('mode must be Train,Val or Test')
    
    if mode=='Train':
        dataset = datasets.ImageFolder(
                                        datapath,
                                        transforms.Compose(
                                                            [
                                                             transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(
                                                                                  mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225]
                                                                                  )
                                                            ]
                                                           )
                                       )
    else:
        dataset = datasets.ImageFolder(
                                        datapath,
                                        transforms.Compose(
                                                            [
                                                             transforms.Resize(256),
                                                             transforms.CenterCrop(224),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(
                                                                                  mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225]
                                                                                  )
                                                            ]
                                                           )
                                       )
    
    if useDistributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    
    dataloader = torch.utils.data.DataLoader(
                                              dataset,
                                              batch_size=BatchSize,
                                              shuffle=(sampler is None) if mode=='Train' else False,
                                              num_workers=Workers,
                                              pin_memory=True,
                                              sampler=sampler if mode=='Train' else None
                                             )
        
    return dataloader,sampler

if __name__=='__main__':
    dataloader,sampler = DataLoader(mode='Test')
    for i,(input,target) in enumerate(dataloader):
        print(input.shape)
        print(target)