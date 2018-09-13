import keras
import keras.layers
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from . import Backbone
from . import retinanet
from ..utils.image import preprocess_image


class LeNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return lenet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['lenet5']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def lenet_retinanet(num_classes, backbone='lenet5', inputs=None, modifier=None, **kwargs):
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    if backbone == 'lenet5':
        backbone = lenet5(inputs, include_top=False)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # get last conv layer from the end of each dense block
    layer_outputs = [backbone.get_layer(name='conv2d_{}'.format(idx+1)).output for idx in range(3)]

    # create the densenet backbone
    backbone = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=backbone.name)

    # invoke modifier if given
    if modifier:
        backbone = modifier(backbone)

    # create the full model
    model = retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone.outputs, **kwargs)

    return model


def lenet5(inputs, include_top=True, *args, **kwargs):

    x = keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, activation='relu')(inputs)
    x = keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

    x = keras.layers.Dense(units=120, activation='relu')(x)
    x = keras.layers.Dense(units=84, activation='relu')(x)

    if include_top:
        x = keras.layers.Dense(units=10, activation='softmax')(x)

    return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
