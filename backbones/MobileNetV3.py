from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)

def _batch_normalization_layer(inputs, momentum=0.997, epsilon=1e-3, training=True, name='bn', reuse=None):
    return tf.layers.batch_normalization(inputs=inputs,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         center=True,
                                         training=training,
                                         name=name,
                                         reuse=reuse)


def _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=False, strides=1, reuse=None, padding="SAME"):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
        padding=padding, #('SAME' if strides == 1 else 'VALID'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name,
        reuse=reuse)
    return conv


def _conv_1x1_bn(inputs, filters_num, name, use_bias=True, training=True, reuse=None):
    kernel_size = 1
    strides = 1
    x = _conv2d_layer(inputs, filters_num, kernel_size, name=name + "/conv", use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, training=training, name=name + '/bn',
                                   reuse=reuse)
    return x


def _conv_bn_relu(inputs, filters_num, kernel_size, name, use_bias=True, strides=1, training=True, activation=relu6, reuse=None):
    x = _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, training=training, name=name + '/bn',reuse=reuse)
    x = activation(x)
    return x


def _dwise_conv(inputs, k_h=3, k_w=3, depth_multiplier=1, strides=(1, 1),
                padding='SAME', name='dwise_conv', use_bias=False,
                reuse=None):
    kernel_size = (k_w, k_h)
    in_channel = inputs.get_shape().as_list()[-1]
    filters = int(in_channel*depth_multiplier)
    return tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                      strides=strides, padding=padding,
                                      data_format='channels_last', dilation_rate=(1, 1),
                                      depth_multiplier=depth_multiplier, activation=None,
                                      use_bias=use_bias, name=name, reuse=reuse)


def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid


def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, reuse=None):
    return tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                           name=name, reuse=reuse)


def _global_avg(inputs, pool_size, strides, padding='valid', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)


def _squeeze_excitation_layer(inputs, out_dim, ratio, layer_name, training=True, reuse=None):
    with tf.variable_scope(layer_name, reuse=reuse):
        # squeeze = _global_avg(inputs, pool_size=inputs.get_shape()[1:-1], strides=1)
        squeeze = _global_avg(inputs, pool_size=2, strides=1)

        excitation = _fully_connected_layer(squeeze, units=out_dim / ratio, name=layer_name + '_excitation1',
                                            reuse=reuse)
        excitation = relu6(excitation)
        excitation = _fully_connected_layer(excitation, units=out_dim, name=layer_name + '_excitation2', reuse=reuse)
        excitation = hard_sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = inputs * excitation
        return scale


def mobilenet_v3_block(inputs, kernel_size, expansion_ratio, output_dim, stride, name, training=True,
                       use_bias=True, shortcut=True, activatation="RE", ratio=16, se=False,
                       reuse=None,**params):
    bottleneck_dim = expansion_ratio

    with tf.variable_scope(name, reuse=reuse):
        # pw mobilenetV2
        net = _conv_1x1_bn(inputs, bottleneck_dim, name="pw", use_bias=use_bias)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # dw
        net = _dwise_conv(net, k_w=kernel_size, k_h=kernel_size, strides=[stride, stride], name='dw',
                          use_bias=use_bias, reuse=reuse)

        net = _batch_normalization_layer(net, momentum=0.997, epsilon=1e-3,
                                         training=training, name='dw_bn', reuse=reuse)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # squeeze and excitation
        if se:
            channel = net.get_shape().as_list()[-1]
            net = _squeeze_excitation_layer(net, out_dim=channel, ratio=ratio, layer_name='se_block')

        # pw & linear
        net = _conv_1x1_bn(net, output_dim, name="pw_linear", use_bias=use_bias)

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            net += inputs
            net = tf.identity(net, name='block_output')

    return net


def MobileNetV3_block(inputs,in_channels,out_channels,multiplier=1.0,reduction_ratio = 4,exp_size=16,
                      kernel_size=3, stride=1, name="conv1", training=True,
                      use_bias=True, activatation="RE", se=False,reuse=None,**params):
    # input_size = inputs.get_shape().as_list()[1:-1]
    # assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))
    in_channels = _make_divisible(in_channels * multiplier)
    out_channels = _make_divisible(out_channels * multiplier)
    exp_size = _make_divisible(exp_size * multiplier)
    x = mobilenet_v3_block(inputs, kernel_size,
                           expansion_ratio=exp_size,
                           output_dim=out_channels,
                           stride=stride, name=name,
                           training=training,
                           use_bias=use_bias,
                           shortcut=(in_channels == out_channels),
                           activatation=activatation,
                           ratio=reduction_ratio,
                           se=se,
                           reuse=reuse,
                           **params)
    return x

def MobileNetV3_backbone(inputs,**config):
    params_conv = {
                   'training': config['training'],
                   }
    with tf.variable_scope('init', reuse=None):
        multiplier = 1.0
        init_conv_out = _make_divisible(16 * multiplier)
        x = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=3, name='init',
                          use_bias=False, strides=2, activation=hard_swish,**params_conv)

    with tf.variable_scope('MobileNetV3', reuse=tf.AUTO_REUSE):
        x = MobileNetV3_block(inputs,in_channels=16,out_channels=16,multiplier=1.0,reduction_ratio = 4,exp_size=16,
                      kernel_size=3, stride=2, name="conv1",
                      use_bias=True, activatation="RE", se=True,reuse=None,**params_conv)
        x = MobileNetV3_block(x,in_channels=16,out_channels=24,multiplier=1.0,reduction_ratio = 4,exp_size=72,
                      kernel_size=3, stride=1, name="conv2",
                      use_bias=True, activatation="RE", se=False,reuse=None,**params_conv)
        x = MobileNetV3_block(x,in_channels=24,out_channels=24,multiplier=1.0,reduction_ratio = 4,exp_size=88,
                      kernel_size=3, stride=1, name="conv3",
                      use_bias=True, activatation="RE", se=False,reuse=None,**params_conv)
        x = MobileNetV3_block(x,in_channels=24,out_channels=40,multiplier=1.0,reduction_ratio = 4,exp_size=96,
                      kernel_size=5, stride=2, name="conv4",
                      use_bias=True, activatation="RE", se=True,reuse=None,**params_conv)
        x = MobileNetV3_block(x,in_channels=40,out_channels=40,multiplier=1.0,reduction_ratio = 4,exp_size=240,
                      kernel_size=5, stride=1, name="conv5",
                      use_bias=True, activatation="RE", se=True,reuse=None,**params_conv)
        x = MobileNetV3_block(x,in_channels=40,out_channels=40,multiplier=1.0,reduction_ratio = 4,exp_size=240,
                      kernel_size=5, stride=1, name="conv6",
                      use_bias=True, activatation="HS", se=True,reuse=None,**params_conv)
        x = MobileNetV3_block(x,in_channels=40,out_channels=48,multiplier=1.0,reduction_ratio = 4,exp_size=120,
                      kernel_size=3, stride=1, name="conv7",
                      use_bias=True, activatation="HS", se=True,reuse=None,**params_conv)
        # x = MobileNetV3_block(x,in_channels=36,out_channels=36,multiplier=1.0,reduction_ratio = 4,exp_size=144,
        #               kernel_size=3, stride=1, name="conv1", training=True,
        #               use_bias=True, activatation="RE", se=True,reuse=None)

    return x

