import csv
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from collections import Counter
import logging
from . import dataset_util
import pprint
pp = pprint.PrettyPrinter(indent=4)

logging.basicConfig(level=getattr(logging, 'DEBUG'))

pixel_counter = Counter()
patch_counter = Counter()

###############################################################################
# Base functions
###############################################################################


def _hex2rgb(hex_value):
    h = hex_value.strip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def parse_color_df(filepath):
    """
    Return RGB values for each color in dataframe where the format looks like:
        id	    name	    color	hex
        6710	Background	black	#000000
        ...     ...         ...     ...

    :param filepath:
    :return: set of (line_count, feature, RGB)
    """
    with open(filepath, 'r') as f:
        color_set = []
        rgb_set = []
        csv_reader = csv.reader(f, delimiter=',')
        line_count = []
        line_num = 0
        for line in csv_reader:
            if line[0].startswith('#') or line[0] == 'id':
                continue
            id_value, name, color, hex_value = line
            color_set.append(name)
            rgb_set.append(_hex2rgb(hex_value))
            line_count.append(line_num)
            line_num += 1
        return list(zip(line_count, color_set, rgb_set))

###############################################################################
# Image functions
###############################################################################


def single_to_multimask(pairs, mask):
    mask_list = []
    binary_list = []
    for group in pairs:
        index, term, c = group
        within_box = np.logical_and(
                mask[..., 0] == c[0],
                mask[..., 1] == c[1],
                mask[..., 2] == c[2],
        ).astype(int)
        mask_list.append(within_box)
        binary_list.append(within_box.max())
        if within_box.max() > 0:
            logging.debug('\t\t\t\tFound {} pixels for {}'.format(within_box.sum()/mask.size, term))
            global pixel_counter
            global patch_counter
            pixel_counter[term] += within_box.sum()
            patch_counter[term] += 1
    return mask_list, binary_list


def get_patches(mask, image, patch_size, overlap, pairs):
    """
    Generator that returns a PNG portions of a larger image with each binary encoded PNG of the masks

    :param mask:
    :param image:
    :param patch_size:
    :param overlap:
    :param pairs:
    :return:
    """
    np_mask = np.asarray(mask)
    np_img = np.asarray(image)
    x_max, y_max, _ = mask.shape
    x = 0
    y = 0

    while x + patch_size <= x_max:
        # Make sure we don't go larger than the image
        if x + patch_size > x_max:
            x = x_max - patch_size

        while y + patch_size <= y_max:
            # Make sure we don't go larger than the image
            if y + patch_size > y_max:
                y = y_max - patch_size

            sub_mask = np_mask[x:x + patch_size, y:y + patch_size, 0:3]
            logging.debug('#####################')
            logging.debug('x: {}\ty:{}'.format(x, y))
            # Reclassify the mask into a list of binary encoded PNGs
            mask_pngs, binary_list = single_to_multimask(pairs, sub_mask)

            # If there is no information, skip
            if sum(binary_list) == 0:
                next

            sub_img = np_img[x:x + patch_size, y:y + patch_size, 0:3]
            results = dict()
            results['sub_image'] = Image.fromarray(sub_img)
            results['sub_masks'] = mask_pngs
            results['x'] = x
            results['y'] = y
            results['binary_list'] = binary_list
            y = int(y + patch_size - (patch_size * overlap))
            yield results
        x = int(x + patch_size - (patch_size * overlap))
        y = 0

    print('Number of Pixels Annotated:')
    global pixel_counter
    pp.pprint(pixel_counter)

    print('Number of Patches Annotated:')
    global patch_counter
    pp.pprint(patch_counter)


def code_tfrecords(project_id, image_id, x, y, patch_size, sub_img, sub_masks, binary_list):
    imgbytearr = io.BytesIO()
    sub_img.save(imgbytearr, format='PNG')
    sub_img = imgbytearr.getvalue()

    feature_dict = {
        'image/height': dataset_util.int64_feature(patch_size),
        'image/width': dataset_util.int64_feature(patch_size),
        'image/start_x': dataset_util.int64_feature(x),
        'image/start_y': dataset_util.int64_feature(y),
        'image/project': dataset_util.bytes_feature(project_id.encode('utf8')),
        'image/source': dataset_util.bytes_feature(image_id.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(sub_img),
        'image/masks/found': dataset_util.int64_list_feature(binary_list)  # Shows whether or not a feature is encoded
    }

    encoded_mask_png_list = []
    for m in sub_masks:
        img = Image.fromarray(m)
        output = io.BytesIO()
        img.save(output, format='PNG')
        encoded_mask_png_list.append(output.getvalue())
    feature_dict['image/masks'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict)).SerializeToString()

    return example
