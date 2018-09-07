"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def freeze(model):
    """ Set all layers in a model to non-trainable.

    The weights for these layers will not be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = False
    return model


def unfreeze(model):
    """ Set all layers in a model to trainable.

    The weights for these layers will be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = True
    return model


def unfreeze_from_block(model, block_id):
    """ Set layers in a model to trainable from defined block name.

    The weights for these layers will be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    trainable_flag = False
    for layer in model.layers:
        if trainable_flag:
            layer.trainable = True
        elif layer.name.find(block_id) != -1:
            trainable_flag = True
            layer.trainable = True
    return model
