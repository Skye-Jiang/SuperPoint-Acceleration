import tensorflow as tf
from tensorflow import layers as tfl

#filters卷积过滤器的数量,对应输出的维数--卷积核的数目（即输出的维度）
#kernel_size过滤器的大小,如果为一个整数则宽和高相同,or(核宽，核长)，单个整数，则表示在各个空间维度的相同长度
#name字符串，层的名字。
#data_format：
#channels_last为(batch, height, width, channels)
#channels_first为(batch, channels, height, width)
#kernel_reg卷积核的正则项tf.contrib.layers.l2_regularizer(kernel_reg),l2正则

def vgg_block(inputs,
              filters,
              kernel_size,
              strides,
              name,
              data_format,
              training=False,
              batch_normalization=True,
              kernel_reg=0.,
              **params):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = tfl.conv2d(inputs,
                       filters,
                       kernel_size,
                       strides,
                       name='conv',
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_reg),
                       data_format=data_format,
                       **params) #converlutional layer with l2
        if batch_normalization:
            x = tfl.batch_normalization(x,
                                        training=training,#set it as Ture when training, otherwise (test) set as False
                                        name='bn',
                                        fused=True,
                                        axis=1 if data_format == 'channels_first' else -1)
            #channels_last=(batch, height, width, channels)
    return x

# training: Python boolean indicating whether the layer should behave in training mode or in inference mode.
# training=True: The layer will normalize its inputs using the mean and variance of the current batch of inputs.
# training=False: The layer will normalize its inputs using the mean and variance of its moving statistics, learned during training.

def vgg_backbone(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}


    params_pool = {'padding': 'SAME', 'data_format': config['data_format']}

    with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):

        # x = vgg_block(inputs, 64, 3,strides=(2,2), name='conv1_1', **params_conv)
        x = vgg_block(inputs, 64, 3, strides=(1,1), name='conv1_1', **params_conv)
        x = vgg_block(x, 64, 3,strides=(1,1),name='conv1_2',**params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool1', **params_pool)

        x = vgg_block(x, 64, 3,strides=(1,1), name='conv2_1', **params_conv)
        x = vgg_block(x, 64, 3,strides=(1,1), name='conv2_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool2', **params_pool)

        x = vgg_block(x, 128, 3,strides=(1,1),name= 'conv3_1', **params_conv)
        x = vgg_block(x, 128, 3,strides=(1,1), name='conv3_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool3', **params_pool)

        x = vgg_block(x, 128, 3, strides=(1,1),name='conv4_1', **params_conv)
        x = vgg_block(x, 128, 3, strides=(1,1),name='conv4_2', **params_conv)

    return x
