
import numpy as np
import torch as th
from torch.utils.data import Dataset

def load_mnist_multi(dataset_path, labels_path, bbox_path, max_objects, size=70000, train_pct=0.9,
                     label_order='spatial', randomize_dataset=False, seed=42):
    """Load MNIST Multi into `BBoxDataset`s (train and validation).
    Each example's labels are padded with an extra '10' class so that every example has
    `max_objects` labels.
    E.g. Image has digits 0, 4, 6 and max_objects=3, then labels = [0, 4, 6]
         Image has digits 1, 8    and max_objects=3, then labels = [1, 8, 10]

    `label_order` is 'random' | 'fixed_random' | 'area' | 'spatial' (default)
    `randomize_dataset` randomizes the label order whenever an example is accessed
        (e.g. whenever a mini-batch is sampled) from the output `BBoxDataset`.
        An evaluation of label order invariance should use randomize_dataset=True.

    To generate a dataset without a validation set (e.g. test), use `train_pct`=1.0.
    """
    dataset_np = np.load(dataset_path).items()[0][1][:size]
    labels_np = np.load(labels_path).items()[0][1][:size]
    bbox_np = np.load(bbox_path).items()[0][1][:size]

    # Label ordering.
    if label_order == 'fixed_random':
        rng = np.random.RandomState(seed)
        fixed_random_order = list(rng.permutation(10))
        for i in xrange(labels_np.shape[0]):
            labels_np[i] = np.array(sorted(list(labels_np[i]),
                                           key=lambda x: fixed_random_order.index(x)))
            bbox_np[i] = np.array(sorted(list(bbox_np[i]),
                                         key=lambda x: fixed_random_order.index(x)))
        pass
    elif label_order == 'area':
        for i in xrange(labels_np.shape[0]):
            areas = [np.prod(bbox[2:]-bbox[:2]) for bbox in bbox_np[i]]
            triples = [(labels_np[i][j], bbox_np[i][j], area) for j, area in enumerate(areas)]
            sorted_triples = sorted(triples, key=lambda x: x[2], reverse=True)
            labels_np[i] = np.array([x[0] for x in sorted_triples])
            bbox_np[i] = np.array([x[1] for x in sorted_triples])
        pass
    elif label_order == 'random':
        for i in xrange(labels_np.shape[0]):
            idxs = np.random.permutation(len(labels_np[i]))
            labels_np[i] = labels_np[i][idxs]
            bbox_np[i] = bbox_np[i][idxs]
    else:  # MNIST Multi labels are ordered spatially by default
        pass

    # Pad with extra class.
    labels_np = np.array([np.concatenate([ls, [10] * (max_objects - ls.shape[0])])
                          for ls in labels_np]).astype(int)

    dataset_t, labels_t = _np_to_t(dataset_np, labels_np)
    train_size = int(size * train_pct)
    X_train_t, y_train_t = dataset_t[:train_size], labels_t[:train_size]

    bbox_np = np.array([np.vstack([bbox, np.zeros((max_objects - bbox.shape[0], bbox.shape[1])) - 1])
                        for bbox in bbox_np])
    bbox_train = bbox_np[:train_size]
    trainset = BBoxDataset(X_train_t, y_train_t, bbox_train,
                           stop_class=10, randomize=randomize_dataset)

    validset = None
    if train_pct != 1.0:
        bbox_test = bbox_np[train_size:]
        X_valid_t, y_valid_t = dataset_t[train_size:], labels_t[train_size:]
        validset = BBoxDataset(X_valid_t, y_valid_t, bbox_test,
                               stop_class=10, randomize=randomize_dataset)

    return trainset, validset

def _np_to_t(x, y):
    """Convert numpy array `x` and numpy integer array `y` into torch Tensors"""
    x = th.from_numpy(x).float()
    x.unsqueeze_(1)
    y = th.LongTensor(y)
    return x, y

class BBoxDataset(Dataset):
    def __init__(self, dataset_t, labels_t, bbox_np,
                 randomize=False, input_transform=None, target_transform=None, stop_class=10):
        """Each item is (image, label, bbox) where bbox is a 4-tuple (x_nw, y_nw, x_se, y_se)."""
        self.dataset_t = dataset_t
        self.labels_t = labels_t
        self.bbox_np = bbox_np
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.stop_class = stop_class
        self.randomize = randomize

    def __getitem__(self, index):
        image = self.dataset_t[index]
        label = self.labels_t[index]
        if self.randomize:
            k = (label != self.stop_class).sum()
            if k < label.size(0):
                label = th.cat((label[th.randperm(k)], label[k:]))
            else:
                label = label[th.randperm(k)]

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, self.bbox_np[index]

    def __len__(self):
        return self.dataset_t.size(0)

if __name__ == '__main__':
    train, valid = load_mnist_multi('output/mnist_custom_100_min20_max50_1_4.npz',
                                    'output/mnist_custom_100_min20_max50_1_4_labels.npz',
                                    'output/mnist_custom_100_min20_max50_1_4_bbox.npz',
                                    max_objects=4,
                                    label_order='fixed_random',
                                    size=100)