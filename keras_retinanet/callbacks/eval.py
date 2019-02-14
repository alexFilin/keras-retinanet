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

import keras
from ..utils.eval import evaluate
import time
import numpy as np
import tensorflow as tf


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        metrics,
        generator,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        tensorboard=None,
        weighted_average=False,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            metrics          : Metrics to display. First will be monitored (choices: precision, mAP, retina, pascal, right, left)
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose
        self.metrics = metrics
        self.monitor_metric = metrics[0]

        super(Evaluate, self).__init__()

        self.metrics_writers = {}
        for metric in self.metrics:
            self.metrics_writers[metric] = tf.summary.FileWriter('./logs/{}'.format(metric))

    def on_train_begin(self, logs=None):
        self.times = []
        self.list_average_precisions = (0, [])

    def calc_class_precisions(self, average_precisions, tag):
        # compute per class average precision
        total_instances = []
        precisions = []
        print(tag)
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        mean_ap_weighted = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
        if self.verbose == 1:
            print('{}: {:.4f}'.format(tag, mean_ap))
            print('Weighted {}: {:.4f}'.format(tag, mean_ap_weighted))
        if self.weighted_average:
            return mean_ap_weighted
        else:
            return mean_ap

    def show_on_tensorboard(self, ap, tag, logs, epoch):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = ap
        if tag != 'precision':
            summary_value.tag = 'ap'
            self.metrics_writers[tag].add_summary(summary, epoch)
            if tag == self.monitor_metric:
                logs[tag] = ap
        if tag == 'precision':
            summary_value.tag = 'precision'
            self.metrics_writers[tag].add_summary(summary, epoch)
            logs[tag] = ap

    def on_epoch_end(self, epoch, logs=None):
        start_time = time.time()
        logs = logs or {}
        # run evaluation
        av_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path,
            metrics=self.metrics
        )

        metrics_precisions = {}
        metrics_values = {}
        for metric in self.metrics:
            metrics_precisions[metric] = av_precisions[metric]
            metrics_values[metric] = self.calc_class_precisions(metrics_precisions[metric], metric)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            for metric in self.metrics:
                self.show_on_tensorboard(metrics_values[metric], metric, logs, epoch)

        current_result = (metrics_values[self.monitor_metric], metrics_precisions[self.monitor_metric])

        if current_result[0] > self.list_average_precisions[0]:
            self.list_average_precisions = current_result

        eval_time = time.time() - start_time
        self.times.append(eval_time)
        print('Evaluation time: {}'.format(eval_time))

    def on_train_end(self, logs=None):
        print('Average evaluation time: {}'.format(np.average(self.times)))
        print('Best classification results ({}):'.format(self.monitor_metric))
        for label, (average_precision, num_annotations) in self.list_average_precisions[1].items():
            print('{:.0f} instances of class'.format(num_annotations),
                  self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))