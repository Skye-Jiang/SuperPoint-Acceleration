import tensorflow as tf
# from tensorflow import layers as tfl
from .layers import bacthnorm, depthwise_conv2d,conv2d



def Mobile_block(inputs,
                 name,
                 num_filters,
                 width_multiplier=0.75,#[0,1]--> 1、0.75、0.5、0.25
                 downsample=False,
                 activation=True,
                 training=True,
                 **params):
    num_filters = round(num_filters * width_multiplier)

    stride = 2 if downsample else 1
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        # depthwise conv2d
        dw_conv = depthwise_conv2d(inputs=inputs, name="depthwise_conv", stride=stride,**params)

        # batchnorm
        bn = bacthnorm(dw_conv, "dw_bn",training=training,**params)
        # relu
        relu = tf.nn.relu(bn)
        # pointwise conv2d (1x1)

        pw_conv = conv2d(inputs=relu, name="pointwise_conv",num_filters=num_filters,stride=1,**params)

        # bn
        bn = bacthnorm(pw_conv, "pw_bn",training=training,**params)

        if activation is False:
            x=bn
        else:
            x=tf.nn.relu(bn)

    return x




def Mobile_backbone(inputs,**config):
    params_conv = {
                   'training': config['training'],
                   }
    with tf.variable_scope("MobileNet",reuse=tf.AUTO_REUSE):
        x = Mobile_block(inputs=inputs, num_filters=32, name='ds_conv_1', downsample=True,**params_conv)
        x = Mobile_block(inputs=x,num_filters=32 ,name='ds_conv_2',downsample=False, **params_conv)
        x = Mobile_block(inputs=x, num_filters=32,name='ds_conv_3',downsample=False, **params_conv)

        x = Mobile_block(inputs=x,num_filters=64,name= 'ds_conv_4',downsample=True, **params_conv)
        x = Mobile_block(inputs=x,num_filters=64, name= 'ds_conv_5',downsample=False, **params_conv)

        x = Mobile_block(inputs=x, num_filters=128, name='ds_conv_6',downsample=True, **params_conv)
        x = Mobile_block(inputs=x, num_filters=128, name='ds_conv_7',downsample=False, **params_conv)
        x = Mobile_block(inputs=x, num_filters=128,name='ds_conv_8',downsample=False, **params_conv)


    return x