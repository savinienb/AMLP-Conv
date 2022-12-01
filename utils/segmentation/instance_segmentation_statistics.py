
import SimpleITK as sitk
import numpy as np
import utils.geometry
import utils.sitk_image
import utils.sitk_np
import utils.np_image
import utils.landmark.transform
import utils.segmentation.metrics
import utils.io.image
import utils.io.text
import utils.io.common
from collections import OrderedDict
import os
import csv
import copy


def intersection(gt_label, predicted_label):
    return np.count_nonzero(np.bitwise_and(gt_label, predicted_label))


def union(gt_label, predicted_label):
    return np.count_nonzero(np.bitwise_or(gt_label, predicted_label))


def intersection_over_union(gt_label, predicted_label):
    i = intersection(gt_label, predicted_label)
    u = union(gt_label, predicted_label)
    return i / u


def intersection_over_union_min_half_gt_size(gt_label, predicted_label):
    i = intersection(gt_label, predicted_label)
    gt_size = np.count_nonzero(gt_label)
    if i <= gt_size * 0.5:
        return 0
    u = union(gt_label, predicted_label)
    return i / u


class InstanceSegmentationStatistics(object):
    def __init__(self,
                 output_folder,
                 save_overlap_image=False):
        self.output_folder = output_folder
        self.save_overlap_image = save_overlap_image
        self.metric_values = {}
        self.metric_keys = ['aiou', 'matches', 'iou_min_half_gt', 'iou_min_half_gt_all']
        self.u = 0
        self.c = 0
        self.total_instances = 0
        self.total_matches = 0
        self.not_in_gt = 0

    def add_labels(self, current_id, prediction_labels, groundtruth_labels):
        aiou = self.get_aiou_values(prediction_labels, groundtruth_labels)
        iou_min_half_gt = self.get_iou_min_half_gt_values(prediction_labels, groundtruth_labels)
        matches = self.get_matches_values(prediction_labels, groundtruth_labels)
        current_metric_values = OrderedDict([('aiou', aiou), ('matches', matches), ('iou_min_half_gt', iou_min_half_gt), ('iou_min_half_gt_all', iou_min_half_gt)])
        print(current_metric_values)
        self.metric_values[current_id] = current_metric_values

    def get_metric_mean_list(self, metric_key):
        metric_values_list = [current_metric_values[metric_key] for current_metric_values in self.metric_values.values()]
        metric_mean_list = list(map(lambda x: sum(x) / len(x), zip(*metric_values_list)))
        return metric_mean_list

    def get_metric_list(self, metric_key):
        metric_values_list = [current_metric_values[metric_key] for current_metric_values in self.metric_values.values()]
        metric_list = [y for x in metric_values_list for y in x]
        return metric_list

    def print_metric_summary(self, metric_key, values):
        format_string = '{} mean: {:.2%}'
        if len(values) > 1:
            format_string += ', std : {:.2%}'
        if len(values) > 2:
            format_string += ', classes: ' + ' '.join(['{:.2%}'] * (len(values) - 2))
            print(format_string.format(metric_key, *values))
        else:
            print(format_string.format(metric_key, *values))

    def print_metric_summaries(self, metric_summaries):
        for key, value in metric_summaries.items():
            self.print_metric_summary(key, value)

    def get_metric_summary(self, metric_key):
        if metric_key == 'aiou':
            return [self.c / self.u]
        if metric_key == 'matches':
            return [self.total_instances, self.total_matches, self.not_in_gt]
        if metric_key == 'iou_min_half_gt_all':
            metric_list = self.get_metric_list(metric_key)
            metric_mean_total = np.mean(metric_list)
            metric_std_total = np.std(metric_list)
            return [metric_mean_total, metric_std_total] + metric_list
        metric_mean_list = self.get_metric_mean_list(metric_key)
        if len(metric_mean_list) > 1:
            metric_mean_total = np.mean(metric_mean_list)
            metric_std_total = np.std(metric_mean_list)
            return [metric_mean_total, metric_std_total] + metric_mean_list
        else:
            return metric_mean_list

    def finalize(self):
        for metric_key in self.metric_keys:
            self.save_metric_values(metric_key)

        metric_summaries = OrderedDict()
        for metric_key in self.metric_keys:
            metric_summaries[metric_key] = self.get_metric_summary(metric_key)

        self.print_metric_summaries(metric_summaries)
        self.save_metric_summaries(metric_summaries)

    def get_label_with_max_metric(self, binary_label, other_labels, metric):
        masked_other_labels = other_labels[binary_label]
        labels = np.unique(masked_other_labels)
        max_metric_value = 0
        max_label = 0
        for label in labels:
            if label == 0:
                continue
            other_binary_label = other_labels == label
            metric_value = metric(binary_label, other_binary_label)
            if max_metric_value < metric_value:
                max_metric_value = metric_value
                max_label = label
        return max_label, max_metric_value

    def get_iou_min_half_gt_values(self, predictions, groundtruth):
        current_metric_values = []
        metric = intersection_over_union_min_half_gt_size
        labels = np.unique(groundtruth)
        for label in labels[1:]:
            current_gt_label = groundtruth == label
            max_label, max_metric_value = self.get_label_with_max_metric(current_gt_label, predictions, metric)
            if max_label == 0:
                # no match
                current_metric_values.append(0)
            else:
                current_metric_values.append(max_metric_value)
        return current_metric_values

    def get_aiou_values(self, predictions, groundtruth):
        current_metric_values = []
        metric = intersection_over_union
        labels = np.unique(groundtruth)
        used_predicted_labels = []
        c = 0
        u = 0
        for label in labels:
            if label == 0:
                continue
            current_gt_label = groundtruth == label
            max_label, max_metric_value = self.get_label_with_max_metric(current_gt_label, predictions, metric)
            if max_label == 0:
                # no match
                continue
            else:
                current_max_label_image = predictions == max_label
                c += intersection(current_gt_label, current_max_label_image)
                u += union(current_gt_label, current_max_label_image)
                used_predicted_labels.append(max_label)
                #current_metric_values.append(max_metric_value)
        other_labels = np.unique(predictions)
        for other_label in other_labels:
            if other_label == 0:
                continue
            if other_label not in used_predicted_labels:
                label_size = np.count_nonzero(predictions == other_label)
                u += label_size
        current_metric_values.append(c / u)
        self.c += c
        self.u += u
        return current_metric_values

    def get_matches_values(self, predictions, groundtruth):
        current_metric_values = []
        metric = intersection_over_union
        labels = np.unique(groundtruth)
        used_predicted_labels = []
        total_instances = 0
        total_matches = 0
        not_in_gt = 0
        for label in labels:
            if label == 0:
                continue
            current_gt_label = groundtruth == label
            max_label, max_metric_value = self.get_label_with_max_metric(current_gt_label, predictions, metric)
            if max_label == 0:
                # no match
                total_instances += 1
                continue
            else:
                total_instances += 1
                total_matches += 1
                used_predicted_labels.append(max_label)
                #current_metric_values.append(max_metric_value)
        other_labels = np.unique(predictions)
        for other_label in other_labels:
            if other_label == 0:
                continue
            if other_label not in used_predicted_labels:
                not_in_gt += 1
        current_metric_values.extend([total_instances, total_matches, not_in_gt])
        self.total_instances += total_instances
        self.total_matches += total_matches
        self.not_in_gt += not_in_gt
        return current_metric_values

    def save_metric_values(self, metric_key):
        metric_dict = OrderedDict([(key, value[metric_key]) for key, value in self.metric_values.items()])
        metric_dict = copy.deepcopy(metric_dict)
        num_values = None
        for value in metric_dict.values():
            num_values = len(value)
            if len(value) > 1:
                value.insert(0, sum(value) / len(value))
        header = [metric_key, 'mean'] + list(range(num_values))
        utils.io.text.save_dict_csv(metric_dict, os.path.join(self.output_folder, metric_key + '.csv'), header)

    def save_metric_summaries(self, metric_summaries):
        file_name = os.path.join(self.output_folder, 'summary.csv')
        utils.io.common.create_directories_for_file_name(file_name)
        with open(file_name, 'w') as file:
            writer = csv.writer(file)
            for key, value in metric_summaries.items():
                writer.writerow([key])
                writer.writerow(['mean'] + list(range(len(value) - 1)))
                writer.writerow(value)
