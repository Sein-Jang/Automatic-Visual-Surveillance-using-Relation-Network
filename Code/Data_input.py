"""Data input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import functools


class ImageInput(object):
    """Generates Image input_fn for training or evaluation."""

    def __init__(self, is_training, data_dir, use_bfloat16, transpose_input=True):
        self.is_training = is_training
        # self.use_bfloat16 = use_bfloat16
        self.data_dir = data_dir
        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input

    def set_shapes(self, batch_size, images, objects, labels):
        """Statically set the batch_size dimension."""
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))

        objects.set_shape(objects.get_shape().merge_with(
            tf.TensorShape([batch_size, 53])))

        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size, 4])))

        return images, objects, labels


    def dataset_parser(self, value):
        """Parse an data record from a serialized string Tensor."""
        keys_to_features = {
            'image':tf.VarLenFeature(tf.float32),
            'object':tf.VarLenFeature(tf.float32),
            'label':tf.VarLenFeature(tf.float32)
            }
        parsed = tf.parse_single_example(value, keys_to_features)
        image = tf.reshape(tf.sparse_tensor_to_dense(parsed['image']), [128, 128, 3])
        object = tf.sparse_tensor_to_dense(parsed['object'])
        label = tf.sparse_tensor_to_dense(parsed['label'])

        return image, object, label


    def input_fn(self,params):
        """Input function which provides a single batch for train or eval."""
        if self.data_dir == None:
            tf.logging.info('Using fake input.')
            return self.input_fn_null(params)

        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # tf.contfib.tpu.RunConfig for details.
        batch_size = params['batch_size']

        # Shuffle the filenames to ensure better randomization.
        file_pattern = os.path.join(
            self.data_dir, 'train-*' if self.is_training else 'validation-*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)


        if self.is_training:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        # Read the data from disk in parallel
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=64, sloppy=True))
        dataset = dataset.shuffle(1024)

        # Parse, preprocess, and batch the data in parallel
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                self.dataset_parser, batch_size=batch_size,
                num_parallel_batches=8,
                drop_remainder=True))

        # Assign static batch size dimension
        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        # Perfetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        (image, object, label) = dataset.make_one_shot_iterator().get_next()

        features = {}
        features['image'] = image
        features['object'] = object

        return features, label

    def input_fn_null(self, params):
        """Input function which provides null(black) images."""
        batch_size = params['batch_size']
        dataset = tf.data.Dataset.range(1).repeat().map(self._get_null_input)
        dataset = dataset.prefetch(batch_size)

        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
        if self.transpose_input:
            dataset = dataset.map(
                lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
                num_parallel_calls=8)

        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        dataset = dataset.prefetch(32)
        tf.logging.info('Input dataset: %s', str(datset))
        return dataset

    def _get_null_input(self, _):
        null_image = tf.zeros([128, 128, 3], tf.bfloat16
                              if self.use_bfloat16 else tf.float32)
        return (null_image, tf.constant(0, tf.int32))
