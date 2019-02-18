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

import os

import cv2
import keras
import numpy as np
import progressbar

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import geopandas
import dsel.common as cmn
from dsel.rendering import rendering
from shapely.geometry import Polygon, Point
from geojson import Feature
from rasterio.transform import xy
from functools import partial


def _compute_ap(recall, precision, metrics):
    """ Compute the average precision, given the recall and precision curves. """

    def compute_classical_mAP(mrec, mpre):
        ap = 0
        for i in range(mrec.size - 1):
            ap += ((mpre[i + 1] + mpre[i]) / 2.0) * (mrec[i + 1] - mrec[i])
        return ap

    def compute_left_mAP(mrec, mpre):
        ap = 0
        for i in range(mrec.size - 1):
            ap += mpre[i] * (mrec[i + 1] - mrec[i])
        return ap

    def compute_right_mAP(mrec, mpre):
        ap = 0
        for i in range(1, mrec.size):
            ap += mpre[i] * (mrec[i] - mrec[i - 1])
        return ap

    def compute_retina_mAP(mrec, mpre):
        """ RETINA MAP (using maximization) """
        # compute the precision envelope
        for i in range(mrec.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def compute_pascal_mAP(mrec, mpre):
        """ PASCAL MAP """
        change_indices = np.where(mrec[1:] != mrec[:-1])[0][::-1]

        mpre[change_indices[0]:mpre.size] = np.max(mpre[change_indices[0]:mpre.size])
        for i in range(change_indices.size - 1):
            mpre[change_indices[i + 1]:change_indices[i] + 1] = np.max(
                mpre[change_indices[i + 1]:change_indices[i] + 1])

        sparse_rec = np.linspace(0.0, 1.0, 11, dtype=np.float16)
        ap = mpre[0]
        for i in range(1, len(sparse_rec)):
            index = np.where(mrec <= sparse_rec[i])[0][-1]
            ap += mpre[index]
        ap /= 11

        return ap

    if 'precision' in metrics:
        metrics.remove('precision')

    mrec_min_max, mpre_min_max = recall.copy(), precision.copy()
    mrec_zero_one, mpre_zero_one = np.concatenate(([0.], recall, [1.])), np.concatenate(([0.], precision, [0.]))

    precisions = {}
    metrics_mapping = {'mAP': partial(compute_classical_mAP, mrec=mrec_min_max, mpre=mpre_min_max),
                       'left': partial(compute_left_mAP, mrec=mrec_min_max, mpre=mpre_min_max),
                       'right': partial(compute_right_mAP, mrec=mrec_min_max, mpre=mpre_min_max),
                       'retina': partial(compute_retina_mAP, mrec=mrec_zero_one, mpre=mpre_zero_one),
                       'pascal': partial(compute_pascal_mAP, mrec=mrec_zero_one, mpre=mpre_zero_one)}

    for metric in metrics:
        precisions[metric] = metrics_mapping[metric]()

    return precisions


def _get_geometry_from_bbox(bbox, geom_type, geo_transform):
    """Create two shapely.geometry objects Polygon, Point from bounding box.

    Args:
        bbox : np.ndarray[np.float32]
            The array of bounding box coordinates [x1, y1, x2, y2].
        geom_type : str
            The type of geometry. Any from ['polygon', 'point'].
        geo_transform : tuple(float)
            The GeoTransform params.

    Returns: any[shapely.geometry.Polygon, shapely.geometry.Point]
        The vectorized bounding box or centroid for bbox.

    """
    ulx, uly = xy(geo_transform, bbox[1], bbox[0])
    rdx, rdy = xy(geo_transform, bbox[3], bbox[2])
    polygon = Polygon([(ulx, uly), (rdx, uly), (rdx, rdy), (ulx, rdy), (ulx, uly)])
    if geom_type.lower() == 'polygon':
        return polygon
    elif geom_type.lower() == 'point':
        return Point(polygon.centroid)
    raise ValueError('Unknown geometry type: {}'.format(geom_type))


def _create_vector_from_bboxes(generator, bboxes, labels, scores, transform, geometry_type, crs):
    """Create vector layer with bounding boxes or centroids.

    Args:
        generator : keras_retinanet.preprocessing.csv_generator.CSVGenerator
            The generator used to run images through the model.
        bboxes : np.ndarray[np.ndarray[np.float32]]
            The [N, 4] matrix (x1, y1, x2, y2) with bounding box coordinates.
        labels : np.ndarray[np.int32]
            The array of N labels.
        scores : np.ndarray[np.float32]
            The array of N scores.
        transform : affine.Affine
            Transformation from pixel coordinates to coordinate reference system.
        geometry_type : str
            The type of geometry to save in geojson.
        crs : dict
            Coordinate Reference System.

    """
    features = []
    for box, label, score in zip(bboxes, labels, scores):
        geometry = _get_geometry_from_bbox(box, geometry_type, transform)
        class_name = generator.label_to_name(label)
        features.append(Feature(geometry=geometry, properties={'class': class_name, 'score': float(score),
                                                               'transform': transform._asdict()}))
    return geopandas.GeoDataFrame.from_features(features, crs=crs)


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None,
                    detect_threshold=0.5, geom_types=None, draw_boxes=False, resize_param=1):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
        detect_threshold : Threshold used for determining what detections to draw.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """

    def _batch_preprocessing(ind):
        raw_image = generator.load_image(ind)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        return image, scale

    def _batch_postprocessing(scales, boxes, scores, labels, ind):
        # correct boxes for image scale
        boxes /= scales

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            image = generator.load_image_bgr_with_geo(ind)
            selection = np.where(image_scores > detect_threshold)[0]

            filename = '{}_{}_{}'.format(i, '-'.join(map(generator.label_to_name, image_labels[selection])),
                                         str(np.around(np.mean(image_scores[selection]), decimals=2)))

            if draw_boxes:
                raw_image = image[0]
                raw_image = rendering(raw_image, r_type='CUM_CUT')

                draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
                draw_detections(raw_image, image_boxes[selection], image_scores[selection], image_labels[selection],
                                label_to_name=generator.label_to_name)

                # with rasterio.open(os.path.join(save_path, filename + '.TIF'), 'w', driver='GTiff',
                #                    height=raw_image.shape[0], width=raw_image.shape[1], count=3,
                #                    dtype=str(raw_image.dtype), crs=image[1], transform=image[2]
                #                    ) as new_dataset:
                #     new_dataset.write(raw_image[..., ::-1].transpose([2, 0, 1]))

                # save_np_using_gdal(os.path.join(save_path, filename+'.TIF'), raw_image[:, :, ::-1], geo_info=image[1])
                cv2.imwrite(os.path.join(save_path, filename + '.PNG'), raw_image)

            if geom_types and len(image_boxes[selection]) != 0:
                for g_type, dir_name in zip(geom_types, dir_names):
                    fn = os.path.join(dir_name, filename + '_{}s.geojson'.format(g_type))
                    vector = _create_vector_from_bboxes(generator, image_boxes[selection] / resize_param, image_labels[selection],
                                                        image_scores[selection], image[2], g_type, image[1])
                    cmn.remove_file(fn)
                    cmn.write_gdf_to_geojson(vector, fn)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[ind][label] = image_detections[image_detections[:, -1] == label, :-1]

    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    if geom_types is None:
        geom_types = []

    dir_names = [os.path.join(save_path, '{}s'.format(geom_type)) for geom_type in geom_types]
    map(lambda name: os.makedirs(name) if not os.path.exists(name) else None, dir_names)

    for i in progressbar.progressbar(range(0, generator.size(), generator.batch_size), prefix='Running network: '):
        # collect images indices for each batch
        batch_indices = [j for j in range(i, i + generator.batch_size) if j < generator.size()]

        # batch_preprocessing
        batch_images, batch_scales = zip(*[_batch_preprocessing(j) for j in batch_indices])

        # run network
        batch_boxes, batch_scores, batch_labels = model.predict_on_batch(np.array(batch_images))[:3]

        # batch_postprocessing
        for j, (scales, boxes, scores, labels) in zip(batch_indices, zip(batch_scales, batch_boxes, batch_scores, batch_labels)):
            _batch_postprocessing(np.expand_dims(scales, 0), np.expand_dims(boxes, 0),
                                  np.expand_dims(scores, 0), np.expand_dims(labels, 0), j)

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    resize_param=1,
    save_path=None,
    vector_types=None,
    draw_boxes=None,
    metrics=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, detect_threshold=iou_threshold,
                                         max_detections=max_detections, save_path=save_path,
                                         geom_types=vector_types, draw_boxes=draw_boxes, resize_param=resize_param)
    all_annotations    = _get_annotations(generator)

    if metrics is None:
        metrics = ['mAP', 'retina', 'left', 'right', 'pascal', 'precision']

    metrics_values = {}
    precisions = {}

    average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in progressbar.progressbar(range(generator.size()), prefix='Computing metrics for class {}: '.format(label)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision, metrics[:])

        for metric in metrics:
            metrics_values[metric] = {}
            if metric == 'precision':
                metrics_values[metric][label] = precision.mean(), num_annotations
            else:
                metrics_values[metric][label] = average_precision[metric], num_annotations

    return metrics_values
