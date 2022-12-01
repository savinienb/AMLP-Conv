
import numpy as np
import utils.sitk_np
import utils.sitk_image
import multiprocessing


class MetricBase(object):
    """
    Segmentation metric base class.
    """
    def __init__(self, print_in_percent=False):
        """
        Initializer.
        :param print_in_percent: If true, metric is printed as a percentage.
        """
        self.print_in_percent = print_in_percent

    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Evaluate the metric with the given prediction and groundtruth sitk image. If a metric cannot be calculated,
        e.g. due to missing labels in prediction or groundtruth, np.nan should be returned.
        In the single label case, each pixel/vosel of the sitk images has the integer value of a label.
        In the multilabel case, the sitk images are multi label vector images, where each vector entry
        corresponds to a label image. A pixel/voxel is considered a part of a label when value == 1.
        :param prediction_sitk: The prediction sitk image.
        :param groundtruth_sitk: The groundtruth sitk image.
        :param labels: The labels list.
        :return: The metric values.
        """
        raise NotImplementedError()


class TpFpFnMetricBase(MetricBase):
    """
    Metric class for metrics that are calculated from true positive, false positive, and false negative numbers.
    """
    def __init__(self, multiprocessing_pool_size=0, print_in_percent=True):
        """
        Initializer.
        :param multiprocessing_pool_size: The size of the multiprocessing pool. If 0, do not use multiprocessing.
                                          Greater numbers typically increase performance, but also memory consumption.
        :param print_in_percent: If true, metric is printed as a percentage.
        """
        super(TpFpFnMetricBase, self).__init__(print_in_percent=print_in_percent)
        self.multiprocessing_pool_size = multiprocessing_pool_size

    def calculate_tp_fp_fn(self, prediction_np, groundtruth_np, label):
        """
        Calculate number of true positives, false positives and false negatives.
        :param prediction_np: Prediction np image.
        :param groundtruth_np: Groundtruth np image.
        :param label: The current label.
        :return: Tuple of tp, fp, fn.
        """
        prediction_equals_label = prediction_np == label
        groundtruth_equals_label = groundtruth_np == label
        tp = np.count_nonzero(np.logical_and(prediction_equals_label, groundtruth_equals_label))
        fp = np.count_nonzero(np.logical_and(prediction_equals_label, np.logical_not(groundtruth_equals_label)))
        fn = np.count_nonzero(np.logical_and(np.logical_not(prediction_equals_label), groundtruth_equals_label))
        return tp, fp, fn

    def calculate_tp_fp_fn_lists_with_hist(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Calculate an np.array of tp, fp, fn entries for all labels. First index of the returned array is the label
        index. Uses histogram based calculation that is faster for multiple labels.
        :param prediction_sitk: The prediction sitk image. Each pixel/voxel has the integer value of a label.
        :param groundtruth_sitk: The groundtruth sitk image. Each pixel/voxel has the integer value of a label.
        :param labels: The labels list.
        :return: np.array of (tp, fp, fn) for all labels.
        """
        prediction_np = utils.sitk_np.sitk_to_np_no_copy(prediction_sitk)
        groundtruth_np = utils.sitk_np.sitk_to_np_no_copy(groundtruth_sitk)
        max_label = max(labels) + 1
        indizes_with_valid_labels = (groundtruth_np >= 0) & (groundtruth_np < max_label)
        h = np.bincount((max_label * groundtruth_np[indizes_with_valid_labels].astype(np.uint16)
                         + prediction_np[indizes_with_valid_labels]).flatten(),
                        minlength=max_label ** 2).reshape(max_label, max_label)
        tp = np.diag(h)
        fp = np.sum(h, 0) - tp
        fn = np.sum(h, 1) - tp
        return np.stack([tp[labels], fp[labels], fn[labels]], axis=1)

    def calculate_tp_fp_fn_lists(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Calculate an np.array of tp, fp, fn entries for all labels. First index of the returned array is the label
        index.
        :param prediction_sitk: The prediction sitk image. Each pixel/voxel has the integer value of a label.
        :param groundtruth_sitk: The groundtruth sitk image. Each pixel/voxel has the integer value of a label.
        :param labels: The labels list.
        :return: np.array of (tp, fp, fn) for all labels.
        """
        return self.calculate_tp_fp_fn_lists_with_hist(prediction_sitk, groundtruth_sitk, labels)

        # old (slower) implementation
        # prediction_np = utils.sitk_np.sitk_to_np_no_copy(prediction_sitk)
        # groundtruth_np = utils.sitk_np.sitk_to_np_no_copy(groundtruth_sitk)
        #
        # if self.multiprocessing_pool_size > 0:
        #     pool = multiprocessing.Pool(self.multiprocessing_pool_size)
        #     return np.array(list(pool.starmap(self.calculate_tp_fp_fn, [(prediction_np, groundtruth_np, label) for label in labels])))
        # else:
        #     return np.array([self.calculate_tp_fp_fn(prediction_np, groundtruth_np, label) for label in labels])

    def evaluate_function(self, tp, fp, fn):
        """
        Return the function value for the given tp, fp, fn values.
        :param tp: Number of true positives.
        :param fp: Number of false positives.
        :param fn: Number of false negatives.
        :return: The calculated value.
        """
        raise NotImplementedError()


class TpFpFnMetricPerLabel(TpFpFnMetricBase):
    """
    Metric class for metrics that are calculated from true positive, false positive, and false negative numbers per label.
    """
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Evaluate the metric with the given prediction and groundtruth sitk image.
        Each pixel/vosel of the sitk images has the integer value of a label.
        The tp, fp, fn values are calculated individually by the evaluation function.
        :param prediction_sitk: The prediction sitk image.
        :param groundtruth_sitk: The groundtruth sitk image.
        :param labels: The labels list.
        :return: The metric values.
        """
        tp_fp_fn_list = self.calculate_tp_fp_fn_lists(prediction_sitk, groundtruth_sitk, labels)
        return [self.evaluate_function(tp, fp, fn) for tp, fp, fn in tp_fp_fn_list]


class TpFpFnMetricAllLabels(TpFpFnMetricBase):
    """
    Metric class for metrics that are calculated from true positive, false positive, and false negative numbers for all labels combined.
    """
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Evaluate the metric with the given prediction and groundtruth sitk image.
        Each pixel/vosel of the sitk images has the integer value of a label.
        The tp, fp, fn values are combined by summing up for each label before calculating the evaluation function.
        :param prediction_sitk: The prediction sitk image.
        :param groundtruth_sitk: The groundtruth sitk image.
        :param labels: The labels list.
        :return: The metric values.
        """
        tp_fp_fn_list = self.calculate_tp_fp_fn_lists(prediction_sitk, groundtruth_sitk, labels)
        tp, fp, fn = [sum(i) for i in zip(*tp_fp_fn_list)]
        return [self.evaluate_function(tp, fp, fn)]


class DiceMetric(TpFpFnMetricPerLabel):
    """
    Dice metric. (Also known as F1 score.)
    """
    def evaluate_function(self, tp, fp, fn):
        """
        Return the Dice value for the given tp, fp, fn values.
        :param tp: Number of true positives.
        :param fp: Number of false positives.
        :param fn: Number of false negatives.
        :return: The calculated value.
        """
        return 2 * tp / (2 * tp + fp + fn) if tp + fn > 0 else np.nan


class JaccardMetric(TpFpFnMetricPerLabel):
    """
    Jaccard metric. (Also known as intersection over union.)
    """
    def evaluate_function(self, tp, fp, fn):
        """
        Return the Jaccard value for the given tp, fp, fn values.
        :param tp: Number of true positives.
        :param fp: Number of false positives.
        :param fn: Number of false negatives.
        :return: The calculated value.
        """
        return tp / (tp + fp + fn) if tp + fn > 0 else np.nan


class DiceMetricAllLabels(TpFpFnMetricAllLabels):
    """
    Dice metric for all labels combined.
    """
    def evaluate_function(self, tp, fp, fn):
        """
        Return the Dice value for the given tp, fp, fn values.
        :param tp: Number of true positives.
        :param fp: Number of false positives.
        :param fn: Number of false negatives.
        :return: The calculated value.
        """
        return 2 * tp / (2 * tp + fp + fn) if tp + fn > 0 else np.nan


class JaccardMetricAllLabels(TpFpFnMetricAllLabels):
    """
    Jaccard metric for all labels combined. (Also known as intersection over union.)
    """
    def evaluate_function(self, tp, fp, fn):
        """
        Return the Jaccard value for the given tp, fp, fn values.
        :param tp: Number of true positives.
        :param fp: Number of false positives.
        :param fn: Number of false negatives.
        :return: The calculated value.
        """
        return tp / (tp + fp + fn) if tp + fn > 0 else np.nan


class HausdorffDistanceMetric(MetricBase):
    """
    Hausdorff distance metric.
    """
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Evaluate the Hausdorff distance with the given prediction and groundtruth sitk image.
        Each pixel/vosel of the sitk images has the integer value of a label.
        :param prediction_sitk: The prediction sitk image.
        :param groundtruth_sitk: The groundtruth sitk image.
        :param labels: The labels list.
        :return: The metric values.
        """
        return utils.sitk_image.hausdorff_distances(prediction_sitk, groundtruth_sitk, labels)


class SurfaceDistanceMetric(MetricBase):
    """
    Surface distance metric.
    """
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Evaluate Surface distance metrics with the given prediction and groundtruth sitk image.
        Calculate mean, median, std, and max surface distances.
        Each pixel/vosel of the sitk images has the integer value of a label.
        :param prediction_sitk: The prediction sitk image.
        :param groundtruth_sitk: The groundtruth sitk image.
        :param labels: The labels list.
        :return: The metric values.
        """
        return utils.sitk_image.surface_distances(prediction_sitk, groundtruth_sitk, labels)


class MultilabelTpFpFnMetricBase(MetricBase):
    """
    Multilabel Metric class that are calculated from true positive, false positive, and false negative numbers.
    """
    def __init__(self, print_in_percent=True):
        """
        Initializer.
        :param print_in_percent: If true, metric is printed as a percentage.
        """
        super(MultilabelTpFpFnMetricBase, self).__init__(print_in_percent=print_in_percent)

    def calculate_tp_fp_fn(self, prediction_np, groundtruth_np, label):
        """
        Calculate number of true positives, false positives and false negatives.
        :param prediction_np: Prediction np image.
        :param groundtruth_np: Groundtruth np image.
        :param label: The current label.
        :return: Tuple of tp, fp, fn.
        """
        tp = np.sum(np.logical_and(prediction_np[..., label] == 1, groundtruth_np[..., label] == 1))
        fp = np.sum(np.logical_and(prediction_np[..., label] == 1, groundtruth_np[..., label] != 1))
        fn = np.sum(np.logical_and(prediction_np[..., label] != 1, groundtruth_np[..., label] == 1))
        return tp, fp, fn

    def calculate_tp_fp_fn_scores(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Calculate a list of (tp, fp, fn) tuples for all labels.
        :param prediction_sitk: The prediction sitk vector image. Each vector index has pixel/voxel value that is either 0 or 1.
        :param groundtruth_sitk: The groundtruth sitk vector image. Each vector index has pixel/voxel value that is either 0 or 1.
        :param labels: The labels list.
        :return: List of tuples of (tp, fp, fn) for all labels.
        """
        prediction_np = utils.sitk_np.sitk_to_np_no_copy(prediction_sitk)
        groundtruth_np = utils.sitk_np.sitk_to_np_no_copy(groundtruth_sitk)
        scores = []
        for label in labels:
            tp, fp, fn = self.calculate_tp_fp_fn(prediction_np, groundtruth_np, label)
            current_score = self.evaluate_function(tp, fp, fn)
            scores.append(current_score)
        return scores

    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Evaluate the metric with the given prediction and groundtruth sitk image.
        The sitk images are multi label vector images, where each vector entry
        corresponds to a label image. A pixel/voxel is considered a part of a label when value == 1.
        :param prediction_sitk: The prediction sitk image.
        :param groundtruth_sitk: The groundtruth sitk image.
        :param labels: The labels list.
        :return: The metric values.
        """
        return self.calculate_tp_fp_fn_scores(prediction_sitk, groundtruth_sitk, labels)

    def evaluate_function(self, tp, fp, fn):
        """
        Return the function value for the given tp, fp, fn values.
        :param tp: Number of true positives.
        :param fp: Number of false positives.
        :param fn: Number of false negatives.
        :return: The calculated value.
        """
        raise NotImplementedError()


class MultilabelDiceMetric(MultilabelTpFpFnMetricBase):
    """
    Dice metric for multilabel (vector) images. (Also known as F1 score.)
    """
    def evaluate_function(self, tp, fp, fn):
        """
        Return the Dice value for the given tp, fp, fn values.
        :param tp: Number of true positives.
        :param fp: Number of false positives.
        :param fn: Number of false negatives.
        :return: The calculated value.
        """
        return 2 * tp / (2 * tp + fp + fn) if tp + fn > 0 else np.nan


class MultilabelHausdorffDistanceMetric(MetricBase):
    """
    Hausdorff distance metric for multilabel (vector) images.
    """
    def __call__(self, prediction_sitk, groundtruth_sitk, labels):
        """
        Evaluate the metric with the given prediction and groundtruth sitk image.
        The sitk images are multi label vector images, where each vector entry
        corresponds to a label image. A pixel/voxel is considered a part of a label when value == 1.
        :param prediction_sitk: The prediction sitk image.
        :param groundtruth_sitk: The groundtruth sitk image.
        :param labels: The labels list.
        :return: The metric values.
        """
        return utils.sitk_image.hausdorff_distances(prediction_sitk, groundtruth_sitk, labels, multi_label=True)
