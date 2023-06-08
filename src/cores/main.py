import argparse
import yaml
import os
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR100
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from cores.params import ExperimentParams
from cores.model import SENet18Cifar
from cores.train import train, classification
from cores.eval import evaluate
from cores.utils import data_multistep_preprocessing, get_step_data


def main():
    # load params from the config file from yaml to dataclass
    parser = argparse.ArgumentParser(description='CL2R: Compatible Lifelong Learning Represenations')
    parser.add_argument("--config_path",
                        help="path of the experiment yaml",
                        default=os.path.join(os.getcwd(), "config.yml"),
                        type=str)
    params = parser.parse_args()
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    args = ExperimentParams()
    for k, v in loaded_params.items():
        args.__setattr__(k, v)
    args.yaml_name = os.path.basename(params.config_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current args:\n{vars(args)}")
    
    # dataset
    args.data_path = osp.join(args.root_folder, "data")
    if not osp.exists(args.data_path):
        os.makedirs(args.data_path)
    if not osp.exists(osp.join(args.root_folder, "checkpoints")):
        os.makedirs(osp.join(args.root_folder, "checkpoints"))
        
    print(f"Loading Training Dataset")
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                               (0.2675, 0.2565, 0.2761))
                                            ])
    dataset_train = CIFAR100(args.data_path, train=True, download=True, transform=train_transform)
    args.num_classes = len(np.unique(dataset_train.targets))

    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                            (0.2675, 0.2565, 0.2761))
                                        ])
    dataset_val = CIFAR100(args.data_path, train=False, download=True, transform=val_transform)
    
    data_multistep_preprocessing(args, dataset_train.targets, dataset_val.targets)

    print(f"Starting Training")
    for step in range(args.nsteps):
        sampler_train, sampler_val, classes_in_step = get_step_data(args, step)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train, 
                                  num_workers=args.num_workers, drop_last=True) 
        
        print(f"Task {step+1} Classes in task: {classes_in_step}")

        net = SENet18Cifar(starting_classes=100, 
                           feat_size=99, 
                           device=args.device)        
        print(f"Created model")

        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-04)
        scheduler_lr = MultiStepLR(optimizer, milestones=args.stages, gamma=0.1)
        criterion_cls = nn.CrossEntropyLoss().cuda(args.device)

        val_loader = DataLoader(dataset_val, batch_size=args.batch_size, sampler=sampler_val, 
                                drop_last=False, num_workers=args.num_workers)
            
        best_acc = 0
        print(f"Starting Epoch Loop at task {step + 1}/{args.nsteps}")
        for epoch in range(args.epochs):
            train(args, net, train_loader, optimizer, epoch, criterion_cls, step)
            acc_val = classification(args, net, val_loader, criterion_cls)
            scheduler_lr.step()
            
            if acc_val > best_acc:
                best_acc = acc_val
                print(f"Saving model, {best_acc=:.2f}")
                ckpt_path = osp.join(*(args.root_folder, "checkpoints", f"ckpt_{step}.pt"))
                torch.save(net.state_dict(), ckpt_path)
                
    print(f"Starting Evaluation")
    evaluate(args)


if __name__ == '__main__':
    main()