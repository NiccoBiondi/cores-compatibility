import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import Sampler

import numpy as np
from PIL import Image
from collections import defaultdict as dd


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_features(args, net, loader):
    features = None
    net.eval()
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs[0].cuda(args.device)
            f, _ = net(inputs)
            f = l2_norm(f)
            if features is not None:
                features = torch.cat((features, f), 0)
            else:
                features = f
    
    return features.detach().cpu().numpy()


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


def log_epoch(n_epochs=None, loss=None, acc=None, epoch=None, task=None, time=None, classification=False):
    acc_str = f"Task {task + 1}" if task is not None else f""
    acc_str += f" Epoch [{epoch + 1}]/[{n_epochs}]" if epoch is not None else f""
    acc_str += f"\t Training Loss: {loss:.4f}" if loss is not None else f""
    acc_str += f"\t Training Accuracy: {acc:.2f}" if acc is not None else f""
    acc_str += f"\t Time: {time:.2f}" if time is not None else f""
    if classification:
        acc_str = acc_str.replace("Training", "Classification")   
    print(acc_str)


class ImagesDataset(Dataset):
    def __init__(self, data=None, targets=None, transform=None):
        self.data = data
        self.targets = targets
        self.transform = None if transform is None else transforms.Compose(transform)

    def __getitem__(self, index):
        if isinstance(self.data[index], str):
            x = Image.open(self.data[index]).convert('RGB')
        else:
            if self.transform: 
                x = Image.fromarray(self.data[index].astype(np.uint8))
            else:
                x = self.data[index]

        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
    

def create_pairs(data_path, num_pos_pairs=3000, num_neg_pairs=3000):
    dataset = CIFAR10(data_path, train=False, download=True)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))
                                    ])
    
    data = np.array(dataset.data)
    targets = np.asarray(dataset.targets) 
    
    imgs = []
    labels = []
    c_p = 0
    c_n = 0
    
    while len(labels) < num_neg_pairs+num_neg_pairs:
        id0, id1 = np.random.choice(np.arange(0,len(targets)),2)
        if targets[id0] == targets[id1] and c_p < num_pos_pairs:
            labels.append(True)
            c_p += 1
        elif targets[id0] != targets[id1] and c_n < num_neg_pairs:
            labels.append(False)
            c_n += 1
        else:
            continue
        if isinstance(data[id0], str):
            img0 = Image.open(data[id0]).convert('RGB')
        else:
            img0 = data[id0]
        if isinstance(data[id1], str):
            img1 = Image.open(data[id1]).convert('RGB')
        else: 
            img1 = data[id1]
        img0 = transform(img0)
        img1 = transform(img1)
        imgs.append(torch.unsqueeze(img0,0))
        imgs.append(torch.unsqueeze(img1,0))
        print(f"{c_p+c_n}/{num_neg_pairs+num_pos_pairs} pairs", end="\r")
    
    print(f"{len(labels)}/{num_neg_pairs+num_pos_pairs} pairs")
    data = torch.cat(imgs).detach().numpy()
    targets = np.asarray(labels)

    query_set = ImagesDataset(data[0::2], targets)
    gallery_set = ImagesDataset(data[1::2], targets)

    return query_set, gallery_set


def data_multistep_preprocessing(args, labels, labels_val):
    nclasses = len(np.unique(labels))
    nclasses_per_step = nclasses // args.nsteps
    indices_perm = np.arange(nclasses)
    
    labels2ids = dd(list)
    for l in np.unique(labels):
        labels2ids[l] = np.where(np.isin(labels, l))[0]
    args.labels2ids = np.array(list(labels2ids.values()))

    labels2ids_val = dd(list)
    for l in np.unique(labels_val):
        labels2ids_val[l] = np.where(np.isin(labels_val, l))[0]
    args.labels2ids_val = np.array(list(labels2ids_val.values()))

    indices = []
    for i in range(1, args.nsteps+1):
        if i == args.nsteps:
            indices.append(indices_perm)
        else:
            indices.append(indices_perm[:nclasses_per_step*i])

    args.indices = indices
    args.targets_indices = dd(list) # to store the targets indices for each step
    args.targets_indices_val = dd(list) # to store the targets indices of val dataset for each step


def get_step_data(args, step):
    new_data_ids = args.indices[step]
    old_data_ids = args.indices[step-1] if step > 0 else []
    new_classes_ids = new_data_ids if step == 0 else np.array([i for i in new_data_ids if i not in old_data_ids]) 
    
    new_targets_idx = np.concatenate(args.labels2ids[new_classes_ids]) 
    new_targets_idx_val = np.concatenate(args.labels2ids_val[new_classes_ids]) 
    if step == 0:
        targets_idx = new_targets_idx
        targets_idx_val = new_targets_idx_val 
    else:
        targets_idx = np.concatenate( (args.targets_indices[step-1], new_targets_idx) )
        targets_idx_val = np.concatenate( (args.targets_indices_val[step-1], new_targets_idx_val) )
    args.targets_indices[step] = targets_idx
    args.targets_indices_val[step] = targets_idx_val

    # random shuffle train targets
    np.random.shuffle(targets_idx)
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(targets_idx)
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(targets_idx_val)

    return sampler_train, sampler_val, new_data_ids

