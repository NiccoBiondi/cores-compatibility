import numpy as np
import os.path as osp
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader

from cores.utils import extract_features
from cores.model import SENet18Cifar
from cores.metrics import average_compatibility, average_multimodel_accuracy
from cores.utils import create_pairs


def evaluate(args):

    query_set, gallery_set = create_pairs(data_path=args.data_path)
    query_loader = DataLoader(query_set, batch_size=args.batch_size, 
                              shuffle=False, drop_last=False, 
                              num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, 
                                num_workers=args.num_workers)

    compatibility_matrix = np.zeros((args.nsteps, args.nsteps))
    targets = query_loader.dataset.targets

    for step in range(args.nsteps):
        ckpt_path = osp.join(*(args.root_folder, "checkpoints", f"ckpt_{step}.pt")) 
        net = SENet18Cifar(resume_path=ckpt_path, 
                                         starting_classes=100, 
                                         feat_size=99, 
                                         device=args.device)
        net.eval() 
        query_feat = extract_features(args, net, query_loader)

        for i in range(step+1):
            ckpt_path = osp.join(*(args.root_folder, "checkpoints", f"ckpt_{i}.pt")) 
            previous_net = SENet18Cifar(resume_path=ckpt_path, 
                                         starting_classes=100, 
                                         feat_size=99, 
                                         device=args.device)
            previous_net.eval() 
        
            gallery_feat = extract_features(args, previous_net, gallery_loader)
            acc = verification(query_feat, gallery_feat, targets)
            compatibility_matrix[step][i] = acc

            if i != step:
                acc_str = f'Cross-test accuracy between model at task {step+1} and {i+1}:'
            else:
                acc_str = f'Self-test of model at task {i+1}:'
            print(f'{acc_str} {acc*100:.2f}')

    print(f"Compatibility Matrix:\n{compatibility_matrix}")

    # compatibility metrics
    ac = average_compatibility(matrix=compatibility_matrix)
    am = average_multimodel_accuracy(matrix=compatibility_matrix)

    print(f"Avg. Comp. {ac:.2f}")
    print(f"Avg. Multi-model Acc. {am:.3f}")


"""Copy from [insightface](https://github.com/deepinsight/insightface)"""
def verification(query_feature, gallery_feature, targets):
    thresholds = np.arange(0, 4, 0.001)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, query_feature, gallery_feature, targets)
    return accuracy.mean()


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds = 10, pca = 0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                actual_issame[test_set])

        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc
