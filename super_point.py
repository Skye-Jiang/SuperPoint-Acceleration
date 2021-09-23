import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .backbones.vgg_faster import vggFaster_backbone
from .backbones.MobileNet import Mobile_backbone
from .backbones.ShuffleNet2 import ShuffleNetV2_backbone
from .backbones.MobileNetV3 import MobileNetV3_backbone
from .backbones.MobileNetV2 import MobileNetV2_backbone
from . import utils


class SuperPoint(BaseModel):
    input_spec = {
                'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_first', #when use MobileNet change it to channel_last
            'grid_size': 16, #when vggFaster:16, otherwise 8
            'detection_threshold': 0.4,
            'descriptor_size': 256,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'lambda_d': 250,
            'descriptor_size': 256,
            'positive_margin': 1,
            'negative_margin': 0.2,
            'lambda_loss': 0.0001,
            'nms': 2,#when vggFaster:2, otherwise 4
            'top_k': 0,
            'box_nms': 1, #Set to 1 for box_nms and 0 for spatial_nms
    }

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            #uncomment one of the four to change the backbone
            features = vgg_backbone(image, **config)
            # features=vggFaster_backbone(image, **config)
            # features=Mobile_backbone(image,**config)
            # features = MobileNetV2_backbone(image, **config)

            # features=ShuffleNetV2_backbone(image, **config)# write the ShuffleNetV2_backbone, but because the None input shape, it can not be applied.
            # features = MobileNetV3_backbone(image, **config)# write the MobileNetV3 backbone, but because the it contain the fully connect layer, it can not be applied.
            detections = utils.detector_head(features, **config)
            descriptors = utils.descriptor_head(features, **config)
            return {**detections, **descriptors}

        results = net(inputs['image'])

        if config['training']:
            warped_results = net(inputs['warped']['image'])
            results = {**results, 'warped_results': warped_results,
                       'homography': inputs['warped']['homography']}

        # Apply NMS and get the final prediction
        prob = results['prob']

        if config['nms']:
            if config['box_nms']:
                prob = tf.map_fn(lambda p: utils.box_nms(
                        p, config['nms'], keep_top_k=config['top_k']), prob)
            else:
                prob = tf.map_fn(lambda p: utils.spatial_nms(
                        p, config['nms']), prob)
            results['prob_nms'] = prob

        results['pred'] = tf.to_int32(tf.greater_equal(
            prob, config['detection_threshold']))

        return results

    def _loss(self, outputs, inputs, **config):
        logits = outputs['logits']
        warped_logits = outputs['warped_results']['logits']
        descriptors = outputs['descriptors_raw']
        warped_descriptors = outputs['warped_results']['descriptors_raw']

        # Switch to 'channels last' once and for all
        if config['data_format'] == 'channels_first':
            logits = tf.transpose(logits, [0, 2, 3, 1])
            warped_logits = tf.transpose(warped_logits, [0, 2, 3, 1])
            descriptors = tf.transpose(descriptors, [0, 2, 3, 1])
            warped_descriptors = tf.transpose(warped_descriptors, [0, 2, 3, 1])

        # Compute the loss for the detector head
        detector_loss = utils.detector_loss(
                inputs['keypoint_map'], logits,
                valid_mask=inputs['valid_mask'], **config)
        warped_detector_loss = utils.detector_loss(
                inputs['warped']['keypoint_map'], warped_logits,
                valid_mask=inputs['warped']['valid_mask'], **config)

        # Compute the loss for the descriptor head
        descriptor_loss = utils.descriptor_loss(
                descriptors, warped_descriptors, outputs['homography'],
                valid_mask=inputs['warped']['valid_mask'], **config)

        tf.summary.scalar('detector_loss1', detector_loss)
        tf.summary.scalar('detector_loss2', warped_detector_loss)
        tf.summary.scalar('detector_loss_full', detector_loss + warped_detector_loss)
        tf.summary.scalar('descriptor_loss', config['lambda_loss'] * descriptor_loss)

        loss = (detector_loss + warped_detector_loss
                + config['lambda_loss'] * descriptor_loss)
        return loss

    def _metrics(self, outputs, inputs, **config):
        pred = inputs['valid_mask'] * outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred * labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred * labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}