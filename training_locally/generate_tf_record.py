"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import random
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from shutil import copy

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


classes = []
def autogen_annotations():
    annotations_string = ""
    for idx, c in enumerate(classes, 1):
        annotations_string += "item {\n\t id: %d\n\t name: '%s'\n}\n\n" % (idx, c)
    with open('annotation.pbtxt', 'w') as f:
        f.write(annotations_string)


def class_text_to_int(row_label):
    return 1 + classes.index(row_label)


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(csv_input, image_dir, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    path = os.path.join(os.getcwd(), image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_file)
    print('Successfully created the TFRecords: {}'.format(output_path))

def xml_to_csv(image_list):
    xml_list = []
    for image in image_list:
        _, tail = os.path.split(image)
        xml_file, _ = os.path.splitext(tail)
        full_xml_loc = os.path.join(os.getcwd(), 'annotations', '%s.xml' % xml_file)
        tree = ET.parse(full_xml_loc)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def create_train_val(image_dir):
    num_classes = 0
    image_count = 0
    for class_name in os.listdir(image_dir):
        image_count += len(os.listdir(os.path.join(image_dir, class_name)))
        num_classes += 1
        
    size_train_set = int(image_count * 0.75)
    images_per_class = int(size_train_set / num_classes)
        
    train_files = []
    val_files = []
    
    for class_name in os.listdir(image_dir):
        folder_path = os.path.join(os.getcwd(), image_dir, class_name)
        files = os.listdir(folder_path)
        random.shuffle(files)
        train_files.extend(files[:images_per_class])
        val_files.extend(files[images_per_class:])
        
    xml_train = xml_to_csv(train_files)
    xml_train.to_csv(('train_labels.csv'), index=None)
    xml_train = xml_to_csv(val_files)
    xml_train.to_csv(('val_labels.csv'), index=None)
    print("%d examples used for eval. Make sure you change this in your config file under \'num_examples\'" % len(val_files))    



def main(_):
    IMAGE_DIR = 'images'
    create_train_val(IMAGE_DIR)
    # Copy all the files to a new directory
    ALL_IMAGES_DIR = 'all_images'
    os.mkdir(ALL_IMAGES_DIR)
    for folder in sorted(os.listdir(IMAGE_DIR)):
        classes.append(folder)
        path = os.path.join(os.getcwd(), IMAGE_DIR, folder)
        for image in os.listdir(path):
            copy(os.path.join(path, image), ALL_IMAGES_DIR)
    autogen_annotations()
    IMAGES_DIR = 'all_images'
    create_tf_record('train_labels.csv', ALL_IMAGES_DIR, 'train.record')
    create_tf_record('val_labels.csv', ALL_IMAGES_DIR, 'val.record')




if __name__ == '__main__':
    tf.app.run()