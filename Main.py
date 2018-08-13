import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

from ImagenetDataLoder import DataLoader
from ShuffleNetV2 import ShuffleNetV2 as Model

import Config

import os
import shutil

useGPU = Config.useGPU

useDistributed = Config.useDistributed
DistBackend = Config.DistBackend
DistUrl = Config.DistUrl
WorldSize = Config.WorldSize

BestScore = 0

StartEpoch = Config.StartEpoch
Epoch = Config.Epoch
Lr = Config.Lr
Momentum = Config.Momentum
WeightDecay = Config.WeightDecay
ModelPath = Config.ModelPath
PretrainModelPath = Config.PretrainModelPath
PrintFreq = Config.PrintFreq
Mode = Config.Mode

def Main():

    model = Model(in_channels=3,num_classes=1000,net_scale=1.0)

    model,loss_fn,optimizer = EnvironmentSetup(model)
    
    model,optimizer = LoadParameters(model,optimizer,PretrainModelPath)
    
    train_loader,train_sample = DataLoader(mode='Train')
    val_loader,val_sample = DataLoader(mode='Val')
    test_loader,test_sample = DataLoader(mode='Test')
    
    if Mode=='Val':
        _ = validate_or_test(val_loader,model,loss_fn)
        return
    elif Mode=='Test':
        _ = validate_or_test(test_loader,model,loss_fn)
        return
    else:
        for epoch in range(StartEpoch,Epoch):
            if useDistributed:
                train_sample.set_epoch(epoch)
            adjust_learning_rate(optimizer,epoch)
            
            train(train_loader,model,loss_fn,optimizer,epoch)
            
            top1 = validate_or_test(val_loader,model,loss_fn)
            
            SaveParameters(model,optimizer,epoch,top1)

def train(train_loader,model,loss_fn,optimizer,epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    for i, (input, target) in enumerate(train_loader):
        if (useGPU or useGPU==0) and torch.cuda.is_available():
            if isinstance(useGPU,list):
                input = input.cuda(useGPU[-1], non_blocking=True)
                target = target.cuda(useGPU[-1], non_blocking=True)
            else:
                input = input.cuda(useGPU, non_blocking=True)
                target = target.cuda(useGPU, non_blocking=True)

        output = model(input)
        loss = loss_fn(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % PrintFreq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top_1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'top_5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))
def validate_or_test(loader, model, loss_fn):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            
            if (useGPU or useGPU==0) and torch.cuda.is_available():
                if isinstance(useGPU,list):
                    input = input.cuda(useGPU[-1], non_blocking=True)
                    target = target.cuda(useGPU[-1], non_blocking=True)
                else:
                    input = input.cuda(useGPU, non_blocking=True)
                    target = target.cuda(useGPU, non_blocking=True)

            output = model(input)
            loss = loss_fn(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
            if i % PrintFreq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'top_1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'top_5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(loader), loss=losses, top1=top1, top5=top5))

        print(' * top_1 {top1.avg:.3f} top_5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
def LoadParameters(model,optimizer,path):
    if not os.path.exists(path):
        print('pretrain model file is not exists...')
        return model,optimizer
    else:
        pass
    
    net_dict = model.state_dict()
    optimizer_dict = optimizer.state_dict()
    
    global StartEpoch
    global BestScore
    
    try:
        pretrain = torch.load(path)
    except:
        pretrain = torch.load(path, map_location=lambda storage, loc: storage)
    
    for k, v in pretrain.items():
        try:
            #All model parameters
            if k=='state_dict':
                for keys in v:
                    net_dict.update( { keys:v[keys] } )
                print('load state_dict')
            elif k=='Epoch':
                StartEpoch = v
                print('load Epoch')
            elif k=='BestScore':
                BestScore = v
                print('load BestScore')
            elif k=='optimizer':
                for keys in v:
                    optimizer_dict.update( { keys:v[keys] } )
                print('load optimizer')
            else:
                raise ValueError('should load state_dict')
        except:
            #Only net state_dict
            net_dict.update( { k:v } )
    
    model.load_state_dict(net_dict)
    optimizer.load_state_dict(optimizer_dict)
    return model,optimizer
def SaveParameters(model,optimizer,epoch,score,path=os.path.join(ModelPath,'checkpoint.pth'),only_save_state_dict=False):
    try:
        is_best = score.cpu()>BestScore.cpu()
    except:
        is_best = score>BestScore
    
    if only_save_state_dict:
        torch.save(model.state_dict(),path)
    else:
        state = {
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'Epoch':epoch,
                'BestScore':max(score,BestScore)
                }
        torch.save(state,path)
    
    if is_best:
        print('save new best model')
        shutil.copyfile(path, os.path.join(ModelPath,'model_best.pth'))
def EnvironmentSetup(model):
    if (useGPU or useGPU==0) and torch.cuda.is_available():
        if isinstance(useGPU,list):
            print('use DataParallel by GPU {}'.format(useGPU))
            model = torch.nn.DataParallel(model.cuda(useGPU[-1]),device_ids=useGPU)
            loss_fn = nn.CrossEntropyLoss().cuda(useGPU[-1])
        else:
            print('use GPU {}'.format(useGPU))
            model = model.cuda(useGPU)
            loss_fn = nn.CrossEntropyLoss().cuda(useGPU)
    elif useDistributed:
        print('use Distributed')
        dist.init_process_group(backend=DistBackend,init_method=DistUrl,world_size=WorldSize)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda())
        loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        print('use CPU, Low Bitch')
        model = model
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(),Lr,momentum=Momentum,weight_decay=WeightDecay)
    
    return model,loss_fn,optimizer
def adjust_learning_rate(optimizer,epoch):
    lr = Lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__=='__main__':
    Main()