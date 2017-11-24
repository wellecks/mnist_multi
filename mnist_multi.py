"""Generates arbitrary sized MNIST with variable number of digits and digit sizes in each image.

python mnist_multi.py --min-digits 1 --max-digits 4 --min-digit-size 20 --max-digit-size 50 --tag my_mnist
"""
import argparse
import cPickle
import cv2
import errno
import gzip
import os
import urllib2
from pprint import pprint
from progressbar import ProgressBar
from scipy import ndimage
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='mnist')
parser.add_argument('--output-dir', default='output')
parser.add_argument('--tag', default='')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--min-digits', type=int, default=2)
parser.add_argument('--max-digits', type=int, default=2)
parser.add_argument('--min-digit-size', type=int, default=28)
parser.add_argument('--max-digit-size', type=int, default=28)
parser.add_argument('--min-num-clutter', type=int, default=8)
parser.add_argument('--max-num-clutter', type=int, default=8)
parser.add_argument('--clutter-width', type=int, default=5)
parser.add_argument('--img-width', type=int, default=100)
parser.add_argument('--dataset-size', type=int, default=70000)
parser.add_argument('--set', action='store_true', help='No duplicate labels per image when set')
parser.add_argument('--num-classes', type=int, default=10, help='Only works with args.set')
args = parser.parse_args()
opts = args.__dict__.copy()
pprint(opts)
np.random.seed(opts['seed'])

def mkdir_p(path, log=True):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    if log:
        print('Created directory %s' % path)

def load_mnist(path, include_validation_set=False):
    file_path = os.path.join(path, 'mnist.pkl.gz')
    if not os.path.isfile(file_path):
        if not os.path.exists(path):
            os.makedirs(path)

        print('Downloading mnist to %s' % (file_path))
        MNIST_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        gz = urllib2.urlopen(MNIST_URL)
        with open(os.path.join(path, 'mnist.pkl.gz'), 'wb') as f:
            f.write(gz.read())

    with gzip.open(os.path.join(path, 'mnist.pkl.gz')) as f:
        train, valid, test = cPickle.load(f)

    if include_validation_set:
        return train, valid, test
    else:
        train = (np.vstack([train[0], valid[0]]),
                 np.hstack([train[1], valid[1]]))
        return train, test

def numbered_partitions(sizes):
    partitions = []
    i = 0
    for size in sizes:
        data = []
        for _ in xrange(size):
            data.append(i)
            i += 1
        partitions.append(data)
    return partitions

def generate_digit(image, width):
    return cv2.resize(image, (width, width))

def compute_bbox(image, xstart, ystart, xend, yend):
    image = image[xstart:xend, ystart:yend]
    # Segment and find connected components
    mask = image > image.mean()
    components, num_components = ndimage.label(mask)

    # Choose largest component (assume digit is larger than any clutter in the area)
    slices = ndimage.find_objects(components)
    areas = [(x.stop - x.start)*(y.stop - y.start) for x, y in slices]
    slice_x, slice_y = slices[np.argmax(areas)]

    bbox_coords = [xstart+slice_x.start,
                   ystart+slice_y.start,
                   xstart+slice_x.stop,
                   ystart+slice_y.stop]
    return bbox_coords

def place_elements(canvas, elements, patch_size, bbox=True):
    # Divide the board into several patches and select a path for each element.
    patch_num = canvas.shape[0] // patch_size
    patch_assignments = np.random.choice(patch_num ** 2, elements.shape[0], replace=False)
    # (x_topleft, y_topleft, x_bottomright, y_bottomright)
    bbox_coords = np.zeros((elements.shape[0], 4))
    for i in range(elements.shape[0]):
        p = patch_assignments[i]
        x_patch = p // patch_num
        y_patch = p % patch_num

        # Top left corner of patch
        xstart = x_patch * patch_size
        ystart = y_patch * patch_size

        # Add random offset within patch
        xoff, yoff = np.random.choice(patch_size-elements[i].shape[0]+1, size=2)
        xstart += xoff
        ystart += yoff

        # Place the element.
        # NOTE(wellecks) Only places non-zero items of each element, which is not always ideal.
        mask = elements[i] > 0
        xend = xstart+elements[i].shape[0]
        yend = ystart+elements[i].shape[1]
        canvas[xstart:xend, ystart:yend][mask == 1] = elements[i][mask == 1]
        if bbox:
            bbox_coords[i] = compute_bbox(canvas, xstart, ystart, xend, yend)
    # Indices of `elements` in order of grid placement (L->R, Top->Bottom order)
    order = np.argsort(patch_assignments)
    bbox_coords = bbox_coords[order]
    if not bbox:
        return canvas, order
    return canvas, order, bbox_coords

def place_digits(canvas, digits):
    # Grid elements equal to max digit size
    # NOTE(wellecks) Not ideal when smallest digit is much smaller than largest digit due
    # to large patch sizes and only having one digit per patch.
    max_digit_size = np.max([d.shape[0] for d in digits])
    return place_elements(canvas, digits, max_digit_size)

def generate_clutter(num_clutter, clutter_width, examples):
    examples = examples[np.random.choice(examples.shape[0], size=num_clutter)]
    clutter = np.zeros((num_clutter, clutter_width, clutter_width))
    for i in xrange(num_clutter):
        xstart, ystart = np.random.choice(examples[i].shape[0]-clutter_width+1, size=2)
        clutter[i] = examples[i, xstart:xstart+clutter_width, ystart:ystart+clutter_width]
    return clutter

def place_clutter(canvas, clutter):
    if clutter.shape[0] == 0:
        return canvas
    canvas, _ = place_elements(canvas, clutter, clutter[0].shape[0], bbox=False)
    return canvas

if __name__ == '__main__':
    mkdir_p(args.output_dir)
    (input_dataset, input_labels), _ = load_mnist(opts['data_dir'])
    input_dataset = input_dataset.reshape((-1, 28, 28))

    num_digits = np.random.randint(opts['min_digits'], opts['max_digits']+1, size=opts['dataset_size'])
    num_clutters = np.random.randint(opts['min_num_clutter'], opts['max_num_clutter']+1, size=opts['dataset_size'])
    digit_sizes = np.random.randint(opts['min_digit_size'], opts['max_digit_size']+1, size=num_digits.sum())
    partitions = numbered_partitions(num_digits)

    if args.set:
        label_idxs = [np.argwhere(input_labels == i).ravel() for i in range(args.num_classes)]
        label_choices = np.hstack([np.random.choice(args.num_classes, size=nd, replace=False) for nd in num_digits]).ravel()
        idxs = [np.random.choice(label_idxs[x]) for x in label_choices]
    else:
        idxs = np.random.choice(input_dataset.shape[0], size=num_digits.sum(), replace=True)

    output_labels = np.array([[input_labels[idxs[j]] for j in partitions[i]]
                              for i in xrange(opts['dataset_size'])])
    output_dataset = np.zeros((opts['dataset_size'], opts['img_width'], opts['img_width']))
    output_bbox = []

    pbar = ProgressBar()
    for i in pbar(range(opts['dataset_size'])):
        canvas = np.zeros((opts['img_width'], opts['img_width']))
        digits = np.array([generate_digit(input_dataset[idxs[j]], digit_sizes[j]) for j in partitions[i]])
        clutter = generate_clutter(num_clutters[i], opts['clutter_width'], input_dataset)
        canvas = place_clutter(canvas, clutter)
        canvas, order, bbox_coords = place_digits(canvas, digits)

        # Sort labels according to grid location.
        output_labels[i] = np.array(output_labels[i])[order]
        output_dataset[i] = canvas
        output_bbox.append(bbox_coords)

    output_bbox = np.array(output_bbox)

    print('Writing files...')
    data_file = os.path.join(opts['output_dir'], 'mnist_custom_%d_%s' % (output_dataset.shape[0], opts['tag']))
    label_file = os.path.join(opts['output_dir'], 'mnist_custom_%d_%s_labels' % (output_dataset.shape[0], opts['tag']))
    bbox_file = os.path.join(opts['output_dir'], 'mnist_custom_%d_%s_bbox' % (output_dataset.shape[0], opts['tag']))
    np.savez_compressed(data_file, output_dataset)
    np.savez_compressed(label_file, output_labels)
    np.savez_compressed(bbox_file, output_bbox)
    print('Done.\nData: %s\nLabels: %s\nBounding boxes: %s\n' % (data_file, label_file, bbox_file))
