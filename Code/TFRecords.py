import math
import os
import random
import tarfile
import urllib
import json
import h5py

from absl import flags
import tensorflow as tf
import numpy as np


# convert the dataset into TF-Records
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# IMAGE, OBJECT, LABEL
def _convert_to_example(image, object, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _float_feature(image),
        'object': _float_feature(object),
        'label': _float_feature(label)}))
    return example


def _process_image_files_batch(output_file, data, idx):
    writer = tf.python_io.TFRecordWriter(output_file)
    #raise ValueError(data[idx]['object'].value.astype(float))

    image = (data[idx]['image'].value.astype(float)/255).reshape(-1)
    question = data[idx]['object'].value.astype(float)
    answer = data[idx]['label'].value.astype(float)
    example = _convert_to_example(image, question, answer)
    writer.write(example.SerializeToString())
    writer.close()


def _process_dataset(output_directory, prefix, num_shards, data):
    files = []

    for shard in range(num_shards):
        output_file = os.path.join(
            output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
        _process_image_files_batch(output_file, data, shard)
        tf.logging.info('Finished writing file: %s' % output_file)
        files.append(output_file)
    return files


def convert_to_tf_records(raw_data_dir):
    """Convert the Image dataset into TF-Record dumps."""
    # Get h5py DATA
    dataset = h5py.File('./h5py/od_acc_change/3_data_80.hy', 'r')

    # Shuffle training records to ensure we are distributing classes
    random.seed(4000)
    def make_shuffle_idx(n):
        order = list(range(n))
        random.shuffle(order)
        return order

    training_shuffle_idx = make_shuffle_idx(len(dataset))
    train_idx = training_shuffle_idx[:2880]
    validation_idx = training_shuffle_idx[2880:]

    train = [dataset[str(i)] for i in train_idx]
    validation = [dataset[str(i)] for i in validation_idx]

    # Create training data
    tf.logging.info('Processing the training data.')
    training_records = _process_dataset(raw_data_dir, 'train', 2880, train)

    # Create validation data
    tf.logging.info('Processing the validation data.')
    validation_records = _process_dataset(raw_data_dir, 'validation', 720, validation)

    return training_records, validation_records



def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    training_records, validation_records = convert_to_tf_records('../Dataset/od_80/')

if __name__ == '__main__':
    tf.app.run()
