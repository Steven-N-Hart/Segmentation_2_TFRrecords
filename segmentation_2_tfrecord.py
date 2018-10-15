import os
from PIL import Image
import warnings
import numpy as np
import tensorflow as tf
import glob

from Lib.write_labels import write_labels
from Lib.utils import parse_color_df, get_patches, code_tfrecords
import logging
import argparse
from skimage.filters import threshold_otsu

###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Extract patches from image and masks')
parser.add_argument("-i", "--image_dir", dest='image_dir', required=True, help="Path to image files (must end in ["
                                                                               "_img.|_mask].png)")
parser.add_argument("-o", "--output_dir", dest='out_dir', default='tfrecords2', help="Output file "
                                                                                                   "directory")
parser.add_argument("-t", "--tf_prefix", dest='tfrecord_prefix', help="Prefix for TFRecords files", default='skin')
parser.add_argument("-p", "--patch_size", dest='patch_size', help="Patch size", default=500, type=int)
parser.add_argument("-f", "--fraction_overlap", dest='overlap', help="Patch size overlap amount", default=0.1, type=float)
parser.add_argument("-c", "--color_config", dest='color_config', required=True, help="Tab separated file of colors "
                                                                                     "for mask")
parser.add_argument("-l", "--label_map_name", dest='label_map_name', default='labels.txt', help="Labels.txt file")
parser.add_argument("-v", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="INFO",
                    help="Set the logging level")
args = parser.parse_args()

if args.logLevel:
    logging.basicConfig(level=getattr(logging, args.logLevel))
else:
    logging.basicConfig(level=getattr(logging, 'INFO'))


image_files = glob.glob(os.path.join(args.image_dir,"*_img.png"))
color_config = args.color_config
outdir = args.out_dir
label_map_name = os.path.join(outdir, args.label_map_name)
tfrecord_prefix = args.tfrecord_prefix
patch_size = args.patch_size
overlap = args.overlap


Image.warnings.simplefilter('error', Image.DecompressionBombError)
warnings.simplefilter(action='ignore', category=FutureWarning)
Image.MAX_IMAGE_PIXELS = None


###############################################################################
# Initial configuration
###############################################################################

training_img_path = os.path.join(outdir, 'image')
training_mask_path = os.path.join(outdir, 'mask')

if not os.path.exists(outdir):
    os.mkdir(outdir)

if not os.path.exists(training_img_path):
    os.mkdir(training_img_path)
    os.mkdir(training_mask_path)

if os.path.exists(label_map_name):
    os.remove(label_map_name)

###############################################################################
# Main
###############################################################################

# training_writer = tf.python_io.TFRecordWriter(os.path.join(outdir, tfrecord_prefix + '_' + 'train.tfrecords'))
# validation_writer = tf.python_io.TFRecordWriter(os.path.join(outdir, tfrecord_prefix + '_' + 'validation.tfrecords'))
# test_writer = tf.python_io.TFRecordWriter(os.path.join(outdir, tfrecord_prefix + '_' + 'test.tfrecords'))

if __name__ == '__main__':
    for image_path in image_files:
        project_id, image_id, _ = os.path.basename(image_path).split('_', 2)
        #image_path = image_path.replace('_img.png', '_img.jpg')
        mask_path = image_path.replace('_img.png', '_mask.png')

        # Check that both files exist
        if os.path.exists(mask_path) and os.path.exists(image_path):
            pass
        else:
            logging.error('Input Files not found!\nImage:{}\nMask:{}'.format(image_path,mask_path))

        pairs = parse_color_df(color_config)
        write_labels(label_map_name, pairs)
        mask = np.array(Image.open(mask_path))
        image = np.array(Image.open(image_path))
        logging.debug("Evaluating {} {} {} {} {}".format(mask.shape, image.shape, patch_size, overlap, pairs))
        # Get images and masks
        results = get_patches(mask, image, patch_size, overlap, pairs)
        logging.debug('Done with results for image: {}'.format(image_id))
        for r in results:
            record = code_tfrecords(project_id, image_id, r['x'], r['y'], patch_size, r['sub_image'], r['sub_mask'],
                                    r['binary_list'])

            # Skip blank images
            bw = np.asarray(r['sub_image'].convert('L'))
            thresh = threshold_otsu(bw)
            if thresh > 230:
                break

            # print into training
            # training_writer.write(record)
            r['sub_image'].save(os.path.join(training_img_path, str(project_id) + '_' + str(
                image_id) + '_' + str(r['x']) + '_' + str(r['y']) + '_' + str(patch_size) + '.jpg'))
            r['sub_mask'].save(os.path.join(training_mask_path, str(project_id) + '_' + str(
                image_id) + '_' + str(r['x']) + '_' + str(r['y']) + '_' + str(patch_size) + '.png'))

    # training_writer.close()
    # validation_writer.close()
    #test_writer.close()
