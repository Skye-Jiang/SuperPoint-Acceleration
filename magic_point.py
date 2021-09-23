import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .backbones.MobileNet import Mobile_backbone
from .backbones.ShuffleNet2 import ShuffleNetV2_backbone

from .utils import detector_head, detector_loss, box_nms
from .homographies import homography_adaptation



class MagicPoint(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
        'data_format': 'channels_first',
        'kernel_reg': 0.,
        'grid_size':16,
        'detection_threshold': 0.4,
        'homography_adaptation': {'num': 0},
        'nms': 2,
        'top_k': 0
    }

#原始图片320*480
    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)
        image = inputs['image']
        # Model_Switch = 2

        # the features map shape is (32, 128, ?, ?)
        # input[32, 128, None, None]
        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            # if Model_Switch==0 or 1 :
            features = vgg_backbone(image, **config)
            # elif Model_Switch==2 or 3 :
            # features=Mobile_backbone(image,**config)
            # elif Model_Switch==4 or 5 :
            #     features=ShuffleNetV2_backbone(image, **config)

            # print("the features map shape is ", features.get_shape())#(1,128,?,?)
            outputs = detector_head(features, **config)
            return outputs

        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            outputs = homography_adaptation(image, net, config['homography_adaptation'])
        else:
            outputs = net(image)

        prob = outputs['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
                                               min_prob=config['detection_threshold'],
                                               keep_top_k=config['top_k']), prob)
            outputs['prob_nms'] = prob

        pred = tf.to_int32(tf.greater_equal(prob, config['detection_threshold']))

        outputs['pred'] = pred

        return outputs

    def _loss(self, outputs, inputs, **config):
        if config['data_format'] == 'channels_first':
            outputs['logits'] = tf.transpose(outputs['logits'], [0, 2, 3, 1])
        return detector_loss(inputs['keypoint_map'], outputs['logits'],
                             valid_mask=inputs['valid_mask'], **config)

    def _metrics(self, outputs, inputs, **config):
        pred = inputs['valid_mask'] * outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred * labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred * labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}
