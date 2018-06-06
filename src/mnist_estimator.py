#!/usr/bin/env python
"""Script to illustrate usage of tf.estimator.Estimator in TF v1.8"""
import argparse
import tensorflow as tf


PREDICT = tf.estimator.ModeKeys.PREDICT
EVAL = tf.estimator.ModeKeys.EVAL
TRAIN = tf.estimator.ModeKeys.TRAIN


# Setup input args parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-dir', type=str, default='./mnist_training',
    help='Output directory for model and training stats.')
parser.add_argument(
    '--learning-rate', type=float, default=0.002,
    help='Adam learning rate.')
parser.add_argument(
    '--train-steps', type=int, default=5000,
    help='Training steps.')
parser.add_argument(
    '--batch-size', type=int, default=128,
    help='Batch size to be used.')


def main(argv=None):
    """Run the training experiment."""
    # Read parameters and input data
    params = parser.parse_args(argv[1:])
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    config = tf.estimator.RunConfig(
        model_dir=params.job_dir,
        save_summary_steps=100,
        log_step_count_steps=100,
        save_checkpoints_steps=500,
    )
    # Setup the Estimator
    model_estimator = build_estimator(config, params)
    # Setup and start training and validation
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: get_train_inputs(params.batch_size, mnist_train),
        max_steps=params.train_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: get_eval_inputs(params.batch_size, mnist_test),
        steps=None,
        start_delay_secs=10,  # Start evaluating after 10 sec.
        throttle_secs=30  # Evaluate only every 30 sec
    )
    tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)


def build_estimator(config, params):
    """
    Build the estimator based on the given config and params.

    Args:
        config (RunConfig): RunConfig object that defines how to run the Estimator.
        params (object): hyper-parameters (can be argparse object).
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params,
    )


def model_fn(features, labels, mode, params):
    """Model function used in the estimator.

    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (object): hyper-parameters (can be argparse object).

    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    # Define model's architecture
    logits = architecture(features, mode)
    class_predictions = tf.argmax(logits, axis=-1)
    # Setup the estimator according to the phase (Train, eval, predict)
    loss = None
    train_op = None
    eval_metric_ops = {}
    predictions = class_predictions
    # Loss will only be tracked during training or evaluation.
    if mode in (TRAIN, EVAL):
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels, tf.int32),
            logits=logits)
    # Training operator only needed during training.
    if mode == TRAIN:
        train_op = get_train_op_fn(loss, params)
    # Evaluation operator only needed during evaluation
    if mode == EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=class_predictions,
                name='accuracy')
        }
    # Class predictions and probabilities only needed during inference.
    if mode == PREDICT:
        predictions = {
            'classes': class_predictions,
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def architecture(inputs, mode, scope='MnistConvNet'):
    """Return the output operation following the network architecture.

    Args:
        inputs (Tensor): Input Tensor
        mode (ModeKeys): Runtime mode (train, eval, predict)
        scope (str): Name of the scope of the architecture

    Returns:
         Logits output Op for the network.
    """
    with tf.variable_scope(scope):
        inputs = inputs / 255
        input_layer = tf.reshape(inputs, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=20,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=40,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        flatten = tf.reshape(pool2, [-1, 4 * 4 * 40])
        dense1 = tf.layers.dense(inputs=flatten, units=256, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense1, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)
        dense2 = tf.layers.dense(inputs=dropout, units=10)
        return dense2


def get_train_op_fn(loss, params):
    """Get the training Op.

    Args:
         loss (Tensor): Scalar Tensor that represents the loss function.
         params (object): Hyper-parameters (needs to have `learning_rate`)

    Returns:
        Training Op
    """
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return train_op


def get_train_inputs(batch_size, mnist_data):
    """Return the input function to get the training data.

    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data ((array, array): Mnist training data as (inputs, labels).

    Returns:
        DataSet: A tensorflow DataSet object to represent the training input
                 pipeline.
    """
    dataset = tf.data.Dataset.from_tensor_slices(mnist_data)
    dataset = dataset.shuffle(
        buffer_size=1000, reshuffle_each_iteration=True
    ).repeat(count=None).batch(batch_size)
    return dataset


def get_eval_inputs(batch_size, mnist_data):
    """Return the input function to get the validation data.

    Args:
        batch_size (int): Batch size of validation iterator that is returned
                          by the input function.
        mnist_data ((array, array): Mnist test data as (inputs, labels).

    Returns:
        DataSet: A tensorflow DataSet object to represent the validation input
                 pipeline.
    """
    dataset = tf.data.Dataset.from_tensor_slices(mnist_data)
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()

