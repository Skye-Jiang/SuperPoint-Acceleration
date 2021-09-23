import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

UPDATE_OPS_COLLECTION = "_update_ops_"

# create variable
def create_variable(name,
                    shape,
                    initializer,
                    dtype=tf.float32,
                    trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype,
            initializer=initializer, trainable=trainable)

# batchnorm layer
def bacthnorm(inputs, name, epsilon=1e-05, momentum=0.99, training=True):

    inputs_shape = inputs.get_shape().as_list()
    params_shape = inputs_shape[-1:] #
    axis = list(range(len(inputs_shape) - 1))#[0,1,2]


    with tf.variable_scope(name):
        beta = create_variable("beta", params_shape,
                               initializer=tf.zeros_initializer())
        gamma = create_variable("gamma", params_shape,
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_variable("moving_mean", params_shape,
                            initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = create_variable("moving_variance", params_shape,
                            initializer=tf.ones_initializer(), trainable=False)
    if training:
        mean, variance = tf.nn.moments(inputs, axes=axis)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=momentum)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=momentum)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)


# depthwise conv2d layer
def depthwise_conv2d(inputs, name, filter_size=3, channel_multiplier=1, stride=1,data_format='NHWC'):

    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(name):
        filter = create_variable("filter", shape=[filter_size, filter_size,in_channels, channel_multiplier],
                       initializer=tf.truncated_normal_initializer(stddev=0.01))

    return tf.nn.depthwise_conv2d(inputs, filter, strides=[1, stride, stride, 1],padding="SAME", rate=[1, 1],data_format=data_format)

# conv2d layer
def conv2d(inputs, name, num_filters, filter_size=1, stride=1,data_format='NHWC'):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(name):
        filter = create_variable("filter", shape=[filter_size, filter_size,in_channels, num_filters],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))

    return tf.nn.conv2d(inputs, filter, strides=[1, stride, stride, 1],padding="SAME",data_format=data_format)

