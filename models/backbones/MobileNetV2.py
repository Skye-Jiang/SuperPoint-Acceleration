import tensorflow as tf

weight_decay=1e-4

def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, training=True, name='bn'):
    return tf.layers.batch_normalization(x,
                      momentum=momentum,
                      epsilon=epsilon,
                      scale=True,
                      training=training,
                      name=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_block(input, out_dim, k, s, training, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, training=training, name='bn')
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)

def pwise_block(input, output_dim, training, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv_1x1(input, output_dim, bias=bias, name='pwb')
        out=batch_norm(out, training=training, name='bn')
        out=relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def res_block(input, expansion_ratio, output_dim, stride, training, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        output_dim = round(output_dim * 0.25)
        # pw
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, training=training, name='pw_bn')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, training=training, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, training=training, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins=conv_1x1(input, output_dim, name='ex_dim')
                net=ins+net
            else:
                net=input+net

        return net


# def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
#     with tf.name_scope(name), tf.variable_scope(name):
#         in_channel = input.get_shape().as_list()[-1]
#         dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
#                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
#                   initializer=tf.truncated_normal_initializer(stddev=0.02))
# 
#         pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim],
#                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
#                   initializer=tf.truncated_normal_initializer(stddev=0.02))
#         strides = [1,stride, stride,1]
#
#         conv=tf.nn.separable_conv2d(input,dwise_filter,pwise_filter,strides,padding=pad, name=name)
#         if bias:
#             biases = tf.get_variable('bias', [output_dim],initializer=tf.constant_initializer(0.0))
#             conv = tf.nn.bias_add(conv, biases)
#         return conv



# def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
#     paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
#     net = tf.pad(inputs, paddings, mode=mode)
#     return net


def MobileNetV2_backbone(inputs,**config):
    params_conv = {
        'training': config['training'],
    }
    exp = 6  # expansion ratio

    with tf.variable_scope('mobilenetv2',reuse=False):
        # net = conv2d_block(inputs, 32, 3, 2, training, name='conv1_1')  # size/2
        # res_block(input, expansion_ratio, output_dim, stride, training, name, bias=False, shortcut=True):
        x = res_block(inputs, 1, 64, 2, name='res1',**params_conv)
        x = res_block(x, exp, 64, 1,  name='res2',**params_conv)  # size/4
        x = res_block(x, exp, 64, 2, name='res3',**params_conv)

        x = res_block(x, exp, 64, 1, name='res4',**params_conv)  # size/8
        x = res_block(x, exp, 128, 1,  name='res5',**params_conv)
        x = res_block(x, exp, 128, 2,  name='res6',**params_conv)

        x = res_block(x, exp, 128, 1, name='res7',**params_conv)
        x = res_block(x, exp, 128, 1,  name='res8',**params_conv)

        # net = res_block(net, exp, 320, 1, training, name='res8_1', shortcut=False)
        #
        # net = pwise_block(net, 1280, training, name='conv9_1')

        return x