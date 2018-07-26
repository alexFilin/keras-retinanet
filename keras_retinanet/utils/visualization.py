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

import cv2
import numpy as np

from .colors import label_color


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0] + 5, b[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(image, caption, (b[0] + 5, b[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    for a in annotations:
        label   = a[4]
        c       = color if color is not None else label_color(label)
        # caption = '{}'.format(label_to_name(label) if label_to_name else label)
        caption = ""
        draw_caption(image, a, caption)

        draw_box(image, a, color=c)


def rendering(image, prc_min_k=2.0, prc_max=98.0, r_type='STD_DEV_K', nodata_value=list()):
    calc_min, calc_max = calc_min_max(image, prc_min_k, prc_max, r_type, nodata_value)
    contrast_stretching = np.maximum(image, calc_min)
    contrast_stretching = np.minimum(contrast_stretching, calc_max)
    contrast_stretching = (contrast_stretching - calc_min) / (calc_max.astype(np.float32) - calc_min) * 255
    return contrast_stretching.astype(np.uint8)


def calc_min_max(image, prc_min_k=2.0, prc_max=98.0, r_type='STD_DEV_K', nodata_value=list()):
    if nodata_value:
        n_ch = 1
        if len(image.shape) > 2:
            n_ch = image.shape[2]
        if len(nodata_value) != n_ch:
            raise RuntimeError('Nodata value length must match with num channels')
        image_stat = image.astype(np.float32).copy()
        ndv = np.full((len(nodata_value),), np.nan, dtype=np.float32)
        val = np.array(nodata_value).astype(np.float32)
        ind = np.all(image_stat == val, axis=len(nodata_value) - 1)
        image_stat[ind] = ndv
    else:
        image_stat = image
    calc_min = np.array(0)
    calc_max = np.array(0)
    # calculation min max
    if r_type == 'STD_DEV_K':
        mean = np.nanmean(image_stat, axis=(0, 1))
        # std = np.sqrt(np.mean((image - mean) ** 2, axis=(0, 1)))
        std = np.nanstd(image_stat, axis=(0, 1))
        calc_max = (mean + std * prc_min_k)
        calc_min = (mean - std * prc_min_k)
    image_dtype = image.dtype
    if r_type == 'CUM_CUT':
        calc_min, calc_max = np.nanpercentile(image_stat, [prc_min_k, prc_max], axis=(0, 1)).astype(image_dtype)
    if not (image_dtype == np.float32 or image_dtype == np.float64
            or image_dtype == np.int8 or image_dtype == np.int16
            or image_dtype == np.int32):
        calc_min = np.maximum(calc_min, 0).astype(image_dtype)
        calc_max = np.minimum(calc_max, 2 ** (8 * image_dtype.itemsize) - 1).astype(image_dtype)
    calc_dif = calc_max - calc_min
    if calc_dif.any() == 0:
        r_type = 'MIN_MAX'
        print('\x1b[0;31;40m' + 'WARNING: Minimum operation performed!' + '\x1b[0m')
    if r_type == 'MIN_MAX':
        calc_min = np.nanmin(image_stat, axis=(0, 1))
        calc_max = np.nanmax(image_stat, axis=(0, 1))
    return calc_min, calc_max