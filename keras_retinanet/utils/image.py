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

from __future__ import division

import cv2
import keras
import numpy as np
import rasterio
import warnings
from skimage import img_as_ubyte, img_as_uint

from .transform import change_transform_origin


def _read_image(path, bit_depth):
    img = rasterio.open(path, 'r').read()
    if bit_depth == '8' and img.dtype == 'uint16':
        img = img_as_ubyte(img)
    elif bit_depth == '16' and img.dtype == 'uint8':
        img = img_as_uint(img)

    img = img.transpose([1, 2, 0])
    return img.copy()


def _read_image_bgr(img):
    img_rgb = img[..., :3]
    img_bgr = img_rgb[:, :, ::-1]
    return img_bgr.copy()


def read_image_bgr(path, bit_depth):
    img = _read_image(path, bit_depth)
    return _read_image_bgr(img)


def read_image_bgrn(path, bit_depth):
    img = _read_image(path, bit_depth)
    img_bgr = _read_image_bgr(img)
    img_nir = img[..., 3]
    image = np.concatenate((img_bgr, img_nir), axis=2)
    return image.copy()


def read_image_bgr_with_geo(path):
    image_source = rasterio.open(path)
    img = image_source.read().transpose([1, 2, 0])
    image_bgr = _read_image_bgr(img)
    return image_bgr, image_source.crs, image_source.transform


def preprocess_image(x, statistics=None, bit_depth=None, channels=None, preprocess='centering', mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    band_values = ['B', 'G', 'R', 'N']
    imagenet_values = [103.939, 116.779, 123.68]

    x = x.astype(keras.backend.floatx())
    if mode == 'tf':
        if bit_depth == '8':
            x /= 127.5
        elif bit_depth == '16':
            x /= 32767.5
        x -= 1.

    elif mode == 'caffe':
        if statistics:
            if preprocess == 'centering':
                for i, band in enumerate(band_values[:channels]):
                    x[..., i] -= statistics['mean'][bit_depth][band]
            elif preprocess == 'standardization':
                for i, band in enumerate(band_values[:channels]):
                    x[..., i] -= (x[..., i] - statistics['mean'][bit_depth][band]) / statistics['std'][bit_depth][band]
            else:
                raise ValueError('{} statistic method is not implemented!'.format(preprocess))
        else:
            if preprocess != 'centering':
                warnings.warn('\nWARNING! Imagenet values are only for preprocessing by centering!\n')

            if bit_depth == '8':
                for i, value in enumerate(imagenet_values):
                    x[..., i] -= value
            elif bit_depth == '16':
                for i, value in enumerate(imagenet_values):
                    x[..., i] -= value * 256
            else:
                raise ValueError('{} bit depth is not implemented!'.format(preprocess))
    else:
        raise ValueError('{} mode is not implemented!'.format(mode))

    return x


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )
    return output


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale
