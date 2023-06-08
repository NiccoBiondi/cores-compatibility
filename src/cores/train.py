import torch

import time

from cl2r.utils import AverageMeter, log_epoch


def train(args, net, train_loader, optimizer, epoch, criterion_cls, step):
    
    start = time.time()
    
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    net.train()
    for inputs, targets in train_loader:
        
        inputs, targets = inputs.cuda(args.device), targets.cuda(args.device)
        feature, output = net(inputs)
        loss = criterion_cls(output, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), inputs.size(0))

        acc_training = accuracy(output, targets, topk=(1,))
        acc_meter.update(acc_training[0].item(), inputs.size(0))

    end = time.time()
    log_epoch(args.epochs, loss_meter.avg, acc_meter.avg, epoch=epoch, task=step, time=end-start)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def classification(args, net, loader, criterion_cls):
    classification_loss_meter = AverageMeter()
    classification_acc_meter = AverageMeter()
    
    net.eval()
    with torch.no_grad():
        for inputs, targets in loader:

            inputs, targets = inputs.cuda(args.device), targets.cuda(args.device)
            feature, output = net(inputs)
            loss = criterion_cls(output, targets)

            classification_acc = accuracy(output, targets, topk=(1,))
            
            classification_loss_meter.update(loss.item(), inputs.size(0))
            classification_acc_meter.update(classification_acc[0].item(), inputs.size(0))

    log_epoch(loss=classification_loss_meter.avg, acc=classification_acc_meter.avg, classification=True)

    classification_acc = classification_acc_meter.avg

    return classification_acc