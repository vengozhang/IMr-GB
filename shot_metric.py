from collections import defaultdict
import torch.nn as nn
import numpy as np
import torch
from scipy.stats import gmean
def shot_metrics(preds, labels, train_labels, many_shot_thr=10, low_shot_thr=2):  #  change with your data!!!
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    labels = np.array(labels).astype(int)

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)
    shot_dict['many_shot_cn'] = np.sum(many_shot_cnt) 
    shot_dict['median_shot_cn'] = np.sum(median_shot_cnt) 
    shot_dict['low_shot_cn'] = np.sum(low_shot_cnt)

    return shot_dict
def val_metrics(preds, labels, y_train):
    shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), y_train)
    outputs, targets = torch.tensor(np.hstack(preds)),torch.tensor(np.hstack(labels))
    preds, labels = [], []
    losses_all, losses_mse, losses_l1, loss_gmean = [], [], [], []
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')
    preds.extend(outputs.data.cpu().numpy())
    labels.extend(targets.data.cpu().numpy())
    loss_mse = criterion_mse(outputs, targets)
    loss_l1 = criterion_l1(outputs, targets)
    loss_all = criterion_gmean(outputs, targets)
    losses_all.extend(loss_all.cpu().numpy())
    loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
    print(f" * Overall: MSE {loss_mse.mean():.3f}\tL1 {loss_l1.mean():.3f}\tG-Mean {loss_gmean:.3f}")
    print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
      f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
    print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
      f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
    print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
      f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")
    print(f" * Many_shot {shot_dict['many_shot_cn']:.3f}\t"
      f"Median {shot_dict['median_shot_cn']:.3f}\tLow_shot {shot_dict['low_shot_cn']:.3f}")
    
if __name__ == '__main__':
    main()