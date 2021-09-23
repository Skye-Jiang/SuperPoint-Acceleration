import tensorflow as tf

from .layer_shuffleNetV2 import ShuffleNetUnitB, ShuffleNetUnitA

def ShuffleNetV2_block(inputs,
                       name,
                       out_channels,
                       n,
                       num_groups=2,
                       downsample=False,
                       **params):
    stride = (2,2) if downsample else (1,1)
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        x = ShuffleNetUnitB(inputs=inputs,name=name, out_channels=out_channels,stride=stride,num_groups=num_groups,**params)

        for _ in range(n):
            x = ShuffleNetUnitA(inputs=x,name=name, out_channels=out_channels,num_groups=num_groups,**params)

    return x


def ShuffleNetV2_backbone(inputs,**config):
    params_conv = {
                   'training': config['training'],
                   }

    with tf.variable_scope("ShuffleNetV2",reuse=tf.AUTO_REUSE):

        x = ShuffleNetV2_block(inputs=inputs, out_channels=32,n=3, name='conv_1', downsample=True,**params_conv)
        x = ShuffleNetV2_block(inputs=x,out_channels=32,n=1,name='conv_2',downsample=False, **params_conv)
        x = ShuffleNetV2_block(inputs=x, out_channels=32,n=1,name='conv_3',downsample=False, **params_conv)

        x = ShuffleNetV2_block(inputs=x,out_channels=64,n=7,name= 'conv_4',downsample=True, **params_conv)
        x = ShuffleNetV2_block(inputs=x,out_channels=64, n=1,name= 'conv_5',downsample=False, **params_conv)

        x = ShuffleNetV2_block(inputs=x, out_channels=128,n=3, name='conv_6',downsample=True, **params_conv)
        x = ShuffleNetV2_block(inputs=x, out_channels=128, n=1,name='conv_7',downsample=False, **params_conv)
        x = ShuffleNetV2_block(inputs=x, out_channels=128,n=1,name='conv_8',downsample=False, **params_conv)

    return x

