import numpy as np
import tensorflow as tf


# Utilities for layers
def __variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return w


# Summaries for variables
def __variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def __conv2d_p(name, inputs, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], inputs.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(inputs, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out

def __depthwise_conv2d_p(name, inputs, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], inputs.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [inputs.shape[-1]], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(inputs, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out

def conv2d(name, inputs, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=False, batchnorm_enabled=False,  dropout_keep_prob=-1,**params):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        conv_o_b = __conv2d_p(name='conv',inputs=inputs, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                                  padding=padding,
                                  initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b,  epsilon=1e-5,**params)
            if activation is False:
                conv_a = conv_o_bn
            else:
                conv_a = tf.nn.relu(conv_o_bn)
        else:
            if activation is False:
                conv_a = conv_o_b
            else:
                conv_a = tf.nn.relu(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(dropout_with_keep, dropout_no_keep,**params)
        else:
            conv_o_dr = conv_a
        conv_o = conv_o_dr

    return conv_o


def depthwise_conv2d(name, inputs, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0, activation=False,
                     batchnorm_enabled=False,**params):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        conv_o_b = __depthwise_conv2d_p(name='depthwise_conv',inputs=inputs, w=w, kernel_size=kernel_size, padding=padding,
                                            stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, epsilon=1e-5,**params)
            if activation is False :
                conv_a = conv_o_bn
            else:
                conv_a = tf.nn.relu(conv_o_bn)
        else:
            if activation is False:
                conv_a = conv_o_b
            else:
                conv_a = tf.nn.relu(conv_o_b)
    return conv_a


# n, h, w, c = inputs.get_shape().as_list()
# x_reshaped = tf.reshape(inputs, [n, h, w, num_groups, c // num_groups])
# x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
# output = tf.reshape(x_transposed, [n, h, w, c])

def channel_shuffle(inputs,name, num_groups):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        n, h, w, c = inputs.get_shape().as_list()
        expend_dim=c // num_groups
        net=tf.split(inputs, expend_dim, axis=3, name="split")
        chs = []
        for i in range(num_groups):
            for j in range(expend_dim):
                chs.append(net[i + j * num_groups])
        net = tf.concat(chs, axis=3, name="concat")
        # n, h, w, c = inputs.get_shape().as_list()
        # x_reshaped = tf.reshape(inputs,[n, h, w, num_groups, c // num_groups])
        # x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        # output = tf.reshape(x_transposed, [n, h,w, c])
    return net


def ShuffleNetUnitA(inputs,name,out_channels,num_groups=2,**params):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        shortcut, x = tf.split(inputs, 2, axis=-1)

        x = conv2d(inputs=inputs,name="shuffleA_conv1",
                   num_filters=out_channels // 2,
                   kernel_size=(1,1),
                   stride=(1, 1),
                   activation = True,
                   batchnorm_enabled = True,
                   **params)

        x = depthwise_conv2d(inputs=x, name="shuffleA_condw1",
                             w=None, stride=(1, 1),
                             kernel_size=(3,3),
                             activation = False,
                             batchnorm_enabled = True,
                             **params)

        x = conv2d(inputs=x, name="shuffleA_conv2",
                   num_filters=out_channels // 2,
                   stride=(1, 1),
                   kernel_size=(1, 1),
                   activation = True,
                   batchnorm_enabled = True,
                   **params)

        x = tf.concat([shortcut, x], axis=-1)
        x = channel_shuffle(inputs=x,name="channel_conv", num_groups=num_groups)

    return x


def ShuffleNetUnitB(inputs,name, out_channels,num_groups=2,stride=(1,1),**params):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        shortcut = inputs
        in_channels = inputs.shape[-1]
        with tf.variable_scope("ShuffleNetUnitB", reuse=tf.AUTO_REUSE):

            x = conv2d(inputs=inputs,name="shuffleB_conv1",
                       num_filters=out_channels // 2,
                       stride=(1, 1),
                       kernel_size=(1, 1),
                       activation = True,
                       batchnorm_enabled = True,
                       **params)

            x = depthwise_conv2d(inputs=x, name="shuffleB_condw1",
                                 w=None, stride=stride,
                                 activation = False,
                                 kernel_size=(3,3),
                                 batchnorm_enabled = True,
                                 **params)

            x = conv2d(inputs=x, name="shuffleB_conv2",
                       num_filters=out_channels - in_channels,
                       w=None, stride=(1, 1),
                       kernel_size=(1, 1),
                       activation = True,
                       batchnorm_enabled = True,
                       **params)

            shortcut = depthwise_conv2d(inputs=shortcut, name="shuffleB_short_condw1",
                                        w=None, stride=stride,
                                        kernel_size=(3,3),
                                        activation = False,
                                        batchnorm_enabled = True,
                                        **params)

            shortcut = conv2d(inputs=shortcut, name="shuffleB_short_conv1",
                              num_filters=in_channels,
                              w=None, stride=(1, 1),
                              activation = True,
                              batchnorm_enabled = True,
                              **params)

            output = tf.concat([shortcut, x], axis=-1)
            output = channel_shuffle(output,name="channel_depthconv", num_groups=num_groups)

    return output