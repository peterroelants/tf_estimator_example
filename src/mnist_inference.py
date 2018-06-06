#!/usr/bin/env python
"""Script to illustrate inference of a trained tf.estimator.Estimator.

NOTE: This is dependent on mnist_estimator.py which defines the model.
mnist_estimator.py can be found at:
https://gist.github.com/peterroelants/9956ec93a07ca4e9ba5bc415b014bcca
"""
import argparse
import numpy as np
import skimage.io
import tensorflow as tf

from mnist_estimator import model_fn


# Setup input args parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-dir', type=str, default='./mnist_training',
    help='Directory with trained model.')


# MNIST sample images
IMAGE_URLS = [
    'https://i.imgur.com/SdYYBDt.png',  # 0
    'https://i.imgur.com/Wy7mad6.png',  # 1
    'https://i.imgur.com/nhBZndj.png',  # 2
    'https://i.imgur.com/V6XeoWZ.png',  # 3
    'https://i.imgur.com/EdxBM1B.png',  # 4
    'https://i.imgur.com/zWSDIuV.png',  # 5
    'https://i.imgur.com/Y28rZho.png',  # 6
    'https://i.imgur.com/6qsCz2W.png',  # 7
    'https://i.imgur.com/BVorzCP.png',  # 8
    'https://i.imgur.com/vt5Edjb.png',  # 9
]


def infer(argv=None):
    """Run the inference and print the results to stdout."""
    params = parser.parse_args(argv[1:])
    # Initialize the estimator and run the prediction
    model_estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        model_dir=params.job_dir
    )
    result = model_estimator.predict(input_fn=test_inputs)
    for r in result:
        print(r)


def test_inputs():
    """Returns training set as Operations.
    Returns:
        (features, ) Operations that iterate over the test set.
    """
    with tf.name_scope('Test_data'):
        images = tf.constant(load_images(), dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((images,))
        # Return as iteration in batches of 1
        return dataset.batch(1).make_one_shot_iterator().get_next()


def load_images():
    """Load MNIST sample images from the web and return them in an array.

    Returns:
        Numpy array of size (10, 28, 28, 1) with MNIST sample images.
    """
    images = np.zeros((10, 28, 28, 1))
    for idx, url in enumerate(IMAGE_URLS):
        images[idx, :, :, 0] = skimage.io.imread(url)
    return images


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run(main=infer)