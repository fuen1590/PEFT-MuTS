from datasets.cmapss import get_partial_data, DEFAULT_ROOT, DEFAULT_SENSORS, Subset
from train.trainable import set_seed
from torch.utils.data import Dataset

import torch
import datasets.xjtu as bearing
import numpy as np
import os


class OrganizedDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def generate_stored_path(dataset, subset, window_size, step_size, partial_mode, partial_ratio):
    return os.path.join(".", dataset, f"{subset}_{window_size}_{step_size}_{partial_mode}_{partial_ratio}")


def generate_cmapss_dataset(subset: str,
                            window_size=30,
                            step_size=15,
                            mode=['engine', 'rul', 'random'],
                            ratio=[0.5, 0.05, 0.8],
                            seed=None):
    if seed is not None:
        set_seed(seed)
    stored_path = generate_stored_path("cmapss", subset, window_size, step_size, mode, ratio)
    if not os.path.exists(stored_path):
        print("Generating new partial data...")
        while True:
            train, test, _, _ = get_partial_data(DEFAULT_ROOT,
                                                 subset=Subset.__members__[subset],
                                                 window_size=window_size,
                                                 slide_step=step_size,
                                                 sensors=DEFAULT_SENSORS,
                                                 rul_threshold=125,
                                                 label_norm=True,
                                                 val_ratio=0,
                                                 mode=mode,
                                                 ratio=ratio,
                                                 retrain_max_label=True)
            if save_dataset(train, test, val=None, stored_path=stored_path, auto_val=0.2):
                break
    print("Loading stored data...")
    train_data = np.load(os.path.join(stored_path, "train_data.npy"))
    train_label = np.load(os.path.join(stored_path, "train_labels.npy"))
    test_data = np.load(os.path.join(stored_path, "test_data.npy"))
    test_label = np.load(os.path.join(stored_path, "test_labels.npy"))
    val_data = np.load(os.path.join(stored_path, "val_data.npy"))
    val_label = np.load(os.path.join(stored_path, "val_labels.npy"))
    train_set = OrganizedDataset(train_data, train_label)
    test_set = OrganizedDataset(test_data, test_label)
    val_set = OrganizedDataset(val_data, val_label)
    return train_set, test_set, val_set


def generate_xjtu_dataset(subset: str,
                          train_bearings,
                          window_size=8192,
                          step_size=4096,
                          mode=['rul', 'random'],
                          ratio=[0.1, 0.8],
                          seed=None):
    if seed is not None:
        set_seed(seed)
    stored_path = generate_stored_path("bearings", subset, window_size, step_size, mode, ratio)
    if not os.path.exists(stored_path):
        train_set = bearing.XJTU(XJTU_path=bearing.DEFAULT_ROOT,
                                 condition=[bearing.Condition.__members__[subset]],
                                 bearing_indexes=[train_bearings],
                                 start_tokens=[[1] * len(train_bearings)],
                                 end_tokens=[[-1] * len(train_bearings)],
                                 labels_type=bearing.LabelsType.TYPE_P,
                                 window_size=window_size,
                                 step_size=step_size)
        test_bearings = list({1, 2, 3, 4, 5} - set(train_bearings))
        test_set = bearing.XJTU(XJTU_path=bearing.DEFAULT_ROOT,
                                condition=[bearing.Condition.__members__[subset]],
                                bearing_indexes=[test_bearings],
                                start_tokens=[[1] * len(test_bearings)],
                                end_tokens=[[-1] * len(test_bearings)],
                                labels_type=bearing.LabelsType.TYPE_P,
                                window_size=window_size,
                                step_size=step_size)
        print("Generating new partial data...")
        while True:
            train_partial = bearing.XJTUPartial(train_set,
                                                mode=mode,
                                                ratio=ratio,
                                                retain_max_label=True)
            if save_dataset(train_partial, test_set, val=None, stored_path=stored_path, auto_val=0.2):
                break
    print("Loading stored data...")
    train_data = np.load(os.path.join(stored_path, "train_data.npy"))
    train_label = np.load(os.path.join(stored_path, "train_labels.npy"))
    test_data = np.load(os.path.join(stored_path, "test_data.npy"))
    test_label = np.load(os.path.join(stored_path, "test_labels.npy"))
    val_data = np.load(os.path.join(stored_path, "val_data.npy"))
    val_label = np.load(os.path.join(stored_path, "val_labels.npy"))
    train_set = OrganizedDataset(train_data, train_label)
    test_set = OrganizedDataset(test_data, test_label)
    val_set = OrganizedDataset(val_data, val_label)
    return train_set, test_set, val_set



def save_dataset(train, test, val, stored_path, auto_val=0.):
    def loop_data(dataset):
        data, labels = [], []
        for i in range(len(dataset)):
            d, l = dataset[i]
            data.append(d)
            labels.append(l)
        return np.stack(data, axis=0), np.stack(labels, axis=0)

    train_data, train_labels = loop_data(train)
    val_data, val_labels = None, None
    if val is not None:
        val_data, val_labels = loop_data(val)
        np.save(os.path.join(stored_path, "val_data.npy"), val_data)
        np.save(os.path.join(stored_path, "val_labels.npy"), val_labels)
    elif val is None and auto_val > 0:
        val_num = int(auto_val * len(train))
        val_index = np.random.choice(np.arange(len(train)), val_num, replace=False)
        val_data, val_labels = train_data[val_index], train_labels[val_index]
    print(f"Available RUL: {np.unique(train_labels)}")
    print(f"Total Samples: {len(train_labels)}")
    ans = input("Should be stored? y/n")
    if ans == 'y':
        os.makedirs(os.path.join(stored_path), exist_ok=True)
        if val_data is not None:
            np.save(os.path.join(stored_path, "val_data.npy"), val_data)
            np.save(os.path.join(stored_path, "val_labels.npy"), val_labels)
        np.save(os.path.join(stored_path, "train_data.npy"), train_data)
        np.save(os.path.join(stored_path, "train_labels.npy"), train_labels)
        test_data, test_labels = loop_data(test)
        np.save(os.path.join(stored_path, "test_data.npy"), test_data)
        np.save(os.path.join(stored_path, "test_labels.npy"), test_labels)
        print(f"Stored in: {stored_path}")
        return True
    else:
        return False


if __name__ == '__main__':
    # ratios = [[0.1, 1.0, 0.5], [0.1, 0.6, 0.5], [0.1, 0.3, 0.5], [0.1, 0.15, 0.5],
    #           [0.1, 0.08, 0.5], [0.1, 0.05, 0.5], [0.1, 0.03, 0.5]]
    ratios = [[0.1, 0.5]]  # 1% 5% 10%
    for ratio in ratios:
        train, test, val = generate_xjtu_dataset("OP_B", [1],
                                                 mode=['rul', 'random'],
                                                 window_size=1024,
                                                 step_size=32768,
                                                 ratio=ratio)
