import numpy as np
from collections import OrderedDict
import os
import csv
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score, f1_score, \
    confusion_matrix, average_precision_score, precision_recall_curve
import traceback
from utils.io.common import create_directories, create_directories_for_file_name


class MammoStatistics(object):
    def __init__(self,
                 labels,
                 current_train_iter,
                 output_folder,
                 loss_keyword,
                 metrics=None,
                 path_to_birads_label=''):
        self.labels = labels
        self.loss_keyword = loss_keyword
        self.current_train_iter = current_train_iter
        # self.path_to_birads_labels = "C:\\Users\\wasse\\Dropbox\\privat\\uni\\Master\\Masterarbeit\\repo\\" \
        #                              "MedicalDataAugmentationTool_tf2\\bin\\mammo\\input\\labels\\" \
        #                              "gt_birads_combined.csv"
        self.path_to_birads_labels = path_to_birads_label
        self.output_folder = output_folder
        self.all_metric_fns = {}
        self.declare_intrinsic_metric_fns()
        self.used_metric_fns = self.select_metric_fns(metrics)
        if 'uncertainty' in self.loss_keyword:
            self.patient_stats = pd.DataFrame(
                columns=['gt_label', 'predicted_label', 'prediction', 'predicted_probability', 'uncertainty'])
        else:
            self.patient_stats = pd.DataFrame(
                columns=['gt_label', 'predicted_label', 'prediction', 'predicted_probability'])
        self.train_cycle_stats = pd.DataFrame(columns=list(self.used_metric_fns.keys()))

    def set_current_iter(self, current_train_iter):
        self.current_train_iter = current_train_iter

    def get_auroc(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        if gt_label is None or predicted_probability is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_probabilities_list = np.array(list(self.patient_stats.predicted_probability))[:, 1]
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_probability,
                              pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_probabilities_list = np.array(list(predicted_probability))[:, 1]
        fpr, tpr, _ = roc_curve(gt_label_list, predicted_probabilities_list)
        return auc(fpr, tpr)

    def get_f1_score(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        try:
            return f1_score(gt_label_list, predicted_label_list)
        except ValueError:
            return 0.0

    def get_precision_recall_score(self, gt_label=None, predicted_label=None, prediction=None,
                                   predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_probability is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_probabilities_list = np.array(list(self.patient_stats.predicted_probability))[:, 1]
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_probability,
                              pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_probabilities_list = np.array(list(predicted_probability))[:, 1]
        try:
            return average_precision_score(gt_label_list, predicted_probabilities_list)
        except ValueError:
            return 0.0

    def get_accuracy_score(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        try:
            return accuracy_score(gt_label_list, predicted_label_list)
        except ValueError:
            return 0.0

    def get_balanced_accuracy_score(self, gt_label=None, predicted_label=None, prediction=None,
                                    predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        try:
            return balanced_accuracy_score(gt_label_list, predicted_label_list)
        except ValueError:
            return 0.0

    def get_confusion_matrix(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        try:
            return self.calculate_cm_matrix_robust(gt_label_list, predicted_label_list, labels=self.labels)
        except ValueError:
            return 0.0

    def calculate_cm_matrix_robust(self, gt_label_list, predicted_label_list, labels=None):
        if len(gt_label_list) == 0 or len(predicted_label_list) == 0:
            return np.zeros((len(labels), len(labels)))
        else:
            return confusion_matrix(gt_label_list, predicted_label_list, labels=labels)

    def get_sensitivity(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        cm = self.calculate_cm_matrix_robust(gt_label_list, predicted_label_list, labels=self.labels).flatten()
        try:
            return cm[3] / (cm[3] + cm[2]) if cm[3] + cm[2] > 0 else 0.0
        except ValueError:
            return 0.0

    def get_specificity(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        cm = self.calculate_cm_matrix_robust(gt_label_list, predicted_label_list, labels=self.labels).flatten()
        try:
            return cm[0] / (cm[1] + cm[0]) if cm[1] + cm[0] > 0 else 0.0
        except ValueError:
            return 0.0

    def get_tn(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        cm = self.calculate_cm_matrix_robust(gt_label_list, predicted_label_list, labels=self.labels).flatten()
        try:
            return cm[0]
        except ValueError:
            return 0

    def get_fp(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        cm = self.calculate_cm_matrix_robust(gt_label_list, predicted_label_list, labels=self.labels).flatten()
        try:
            return cm[1]
        except ValueError:
            return 0

    def get_fn(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        cm = self.calculate_cm_matrix_robust(gt_label_list, predicted_label_list, labels=self.labels).flatten()
        try:
            return cm[2]
        except ValueError:
            return 0

    def get_tp(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        # [TN, FP]
        # [FN, TP]
        if gt_label is None or predicted_label is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
        cm = self.calculate_cm_matrix_robust(gt_label_list, predicted_label_list, labels=self.labels).flatten()
        try:
            return cm[3]
        except ValueError:
            return 0

    def get_mean_evidence_succ(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        if gt_label is None or predicted_label is None or prediction is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
            prediction_list = list(self.patient_stats.prediction)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(prediction, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
            prediction_list = list(prediction)
        total_evidence_per_patient = np.sum(np.array(prediction_list), axis=1)
        match = np.float32(np.equal(predicted_label_list, gt_label_list))
        mean_ev_succ = np.sum(total_evidence_per_patient * match) / (np.sum(match) + 1e-20)
        return mean_ev_succ

    def get_mean_evidence_fail(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        if gt_label is None or predicted_label is None or prediction is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_label_list = list(self.patient_stats.predicted_label)
            prediction_list = list(self.patient_stats.prediction)
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(prediction, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_label_list = list(predicted_label)
            prediction_list = list(prediction)
        total_evidence_per_patient = np.sum(np.array(prediction_list), axis=1)
        match = np.float32(np.equal(predicted_label_list, gt_label_list))
        mean_ev_fail = np.sum(total_evidence_per_patient * (1 - match)) / (np.sum(np.abs(1 - match)) + 1e-20)
        return mean_ev_fail

    def get_mean_uncertainty(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None):
        if prediction is None:
            prediction_list = list(self.patient_stats.prediction)
        else:
            assert isinstance(prediction, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            prediction_list = list(prediction)
        alpha = np.array(prediction_list) + 1
        if len(alpha) == 0:
            return 0.0
        else:
            uncertainty = 2 / np.sum(alpha, axis=1)
            return np.mean(uncertainty)

    def get_current_train_cycle_stats(self):
        return list(self.train_cycle_stats.loc[self.current_train_iter])

    def get_current_metric_dict(self):
        return OrderedDict(list(zip(list(self.train_cycle_stats.columns), self.get_current_train_cycle_stats())))

    def get_current_metric_dict_for_tf_summary(self, postfix=''):
        return OrderedDict(list(zip(self.get_metric_names_for_tf_summary(postfix),
                                    list(self.train_cycle_stats.loc[self.current_train_iter]))))

    def get_metric_names(self):
        return list(self.used_metric_fns.keys())

    def get_metric_names_for_tf_summary(self, postfix=''):
        tmp = [k + postfix for k in self.used_metric_fns.keys()]
        return tmp

    def get_non_binary_softmax_prediction(self, prediction):
        return scipy.special.softmax(prediction, axis=0)

    def get_EDL_class_probabilites(self, prediction):
        return prediction / sum(prediction)

    def select_metric_fns(self, metrics):
        used_metric_fns = {}
        for metric in metrics:
            if metric in self.all_metric_fns:
                used_metric_fns[metric] = self.all_metric_fns[metric]
            else:
                raise KeyError("Undefined metric: {}".format(metric))
        return used_metric_fns

    def add_patient_labels(self, patient_id, true_label, prediction):
        #predicted_label = int(prediction[0] > 0.0)
        if 'uncertainty' in self.loss_keyword:
            predicted_label = int(np.argmax(prediction))
            self.patient_stats.at[patient_id] = [true_label, predicted_label, prediction,
                                                 self.get_EDL_class_probabilites(prediction),
                                                 len(self.labels) / np.sum(prediction + 1)]
        elif 'sigmoid' in self.loss_keyword:
            predicted_label = int(prediction[0] > 0.5)
            self.patient_stats.at[patient_id] = [true_label, predicted_label, prediction,
                                                 np.stack([1-prediction, prediction], axis=0)]
        elif 'regression' in self.loss_keyword:
            mean = prediction[0]
            # TODO use sigma for neg_log_normal loss
            predicted_label = mean > 0.0
            probability = np.clip((mean / 4) + 0.5, 0, 1)
            self.patient_stats.at[patient_id] = [true_label, predicted_label, mean,
                                                 np.stack([1-probability, probability], axis=0)]
        else:
            predicted_label = int(np.argmax(prediction))
            self.patient_stats.at[patient_id] = [true_label, predicted_label, prediction,
                                                 self.get_non_binary_softmax_prediction(prediction)]

    def declare_intrinsic_metric_fns(self):
        self.all_metric_fns['F1_score'] = self.get_f1_score
        self.all_metric_fns['Accuracy'] = self.get_accuracy_score
        self.all_metric_fns['Balanced_Accuracy'] = self.get_balanced_accuracy_score
        self.all_metric_fns['Confusion_Matrix'] = self.get_confusion_matrix
        self.all_metric_fns['AUROC'] = self.get_auroc
        self.all_metric_fns['Precision_Recall'] = self.get_precision_recall_score
        self.all_metric_fns['Sensitivity'] = self.get_sensitivity
        self.all_metric_fns['Specificity'] = self.get_specificity
        self.all_metric_fns['TN'] = self.get_tn
        self.all_metric_fns['FP'] = self.get_fp
        self.all_metric_fns['FN'] = self.get_fn
        self.all_metric_fns['TP'] = self.get_tp
        self.all_metric_fns['mean_ev_succ'] = self.get_mean_evidence_succ
        self.all_metric_fns['mean_ev_fail'] = self.get_mean_evidence_fail
        self.all_metric_fns['uncertainty'] = self.get_mean_uncertainty

    def calculate_metric_values(self):
        for metric_key, metric in self.used_metric_fns.items():
            try:
                self.train_cycle_stats.loc[self.current_train_iter, metric_key] = \
                    metric(self.patient_stats.gt_label, self.patient_stats.predicted_label,
                           self.patient_stats.prediction, self.patient_stats.predicted_probability)
            except Exception as e:
                print(e)
                self.train_cycle_stats.loc[self.current_train_iter, metric_key] = 0.0

    # ---------------- Print stuff ----------------
    def plot_softmax(self):
        try:
            softmax_pos = []
            softmax_neg = []
            for patient in range(self.patient_stats.shape[0]):
                if self.patient_stats.gt_label.iloc[patient] == 1:
                    softmax_pos.append(self.patient_stats.predicted_probability[patient][1])
                else:
                    softmax_neg.append(self.patient_stats.predicted_probability[patient][1])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(softmax_pos, bins=10, color='red', label='Malignant')
            ax.hist(softmax_neg, bins=10, color='green', alpha=0.4, label='Benign')
            ax.legend()
            ax.set_title('Malignant (positive) prediction probabilities')
            file_name = os.path.join(self.output_folder, 'test', 'softmax_histogram_iter{}.png'
                                     .format(self.current_train_iter))
            create_directories_for_file_name(file_name)
            # print(file_name)
            fig.savefig(file_name)
            plt.close(fig)
        except ValueError as e:
            print(e)
            print("Evidence was too large probably: ")

    def plot_prediction_per_birads(self, output_folder=None):
        if output_folder is not None:
            path = output_folder
        else:
            path = os.path.join(self.output_folder, "test")

        create_directories(path)

        birads_tx_fx_dict = {'0': [0, 0],
                             '1': [0, 0],
                             '2': [0, 0],
                             '3': [0, 0],
                             '4': [0, 0]}
        with open(self.path_to_birads_labels, 'r') as f:
            reader = csv.reader(f)
            for (current_id, birads) in reader:
                try:
                    groundtruth_label = self.patient_stats.loc[current_id, 'gt_label']
                    predicted_label = self.patient_stats.loc[current_id, 'predicted_label']
                    if groundtruth_label == predicted_label:
                        birads_tx_fx_dict[birads][0] += 1
                    else:
                        birads_tx_fx_dict[birads][1] += 1
                except KeyError:
                    continue

        correct_predictions_per_birads = np.array(list(birads_tx_fx_dict.values()))[:, 0]
        incorrect_predictions_per_birads = np.array(list(birads_tx_fx_dict.values()))[:, 1]
        num_birads_classes = len(correct_predictions_per_birads)
        fig, ax = plt.subplots()
        ax.bar(np.arange(1, num_birads_classes + 1) - 0.1, correct_predictions_per_birads, width=0.2, color='g')
        ax.bar(np.arange(1, num_birads_classes + 1) + 0.1, incorrect_predictions_per_birads, width=0.2, color='r')
        ax.set_xlabel("classification per birads")
        ax.set_ylabel("number of patients")
        ax.legend(["correct", "incorrect"])
        plt.xticks(np.arange(1, num_birads_classes + 1),
                   ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5'])
        plt.savefig(os.path.join(path, "predictions_per_birads_iter{}.png".format(self.current_train_iter)))
        plt.close(fig)

    def plot_uncertainty_boxplot_per_cm_group(self, output_folder=None):
        if output_folder is not None:
            path = output_folder
        else:
            path = os.path.join(self.output_folder, "test")

        tp_uncertainty_list = []
        tn_uncertainty_list = []
        fn_uncertainty_list = []
        fp_uncertainty_list = []
        for current_id in self.patient_stats.index.values:
            groundtruth_label = self.patient_stats.loc[current_id, 'gt_label']
            predicted_label = self.patient_stats.loc[current_id, 'predicted_label']
            if groundtruth_label == predicted_label:
                if groundtruth_label == 1:
                    tp_uncertainty_list.append(self.patient_stats.loc[current_id, 'uncertainty'])
                else:
                    tn_uncertainty_list.append(self.patient_stats.loc[current_id, 'uncertainty'])
            else:
                if groundtruth_label == 1:
                    fn_uncertainty_list.append(self.patient_stats.loc[current_id, 'uncertainty'])
                else:
                    fp_uncertainty_list.append(self.patient_stats.loc[current_id, 'uncertainty'])

        file_id_dict = {'tp': tp_uncertainty_list, 'tn': tn_uncertainty_list, 'fp': fp_uncertainty_list,
                        'fn': fn_uncertainty_list}

        fig, ax = plt.subplots()
        ax.boxplot(file_id_dict.values())
        ax.set_title("Uncertainty per group")
        ax.set_ylim([0, 1])
        plt.xticks(range(1, len(file_id_dict.keys()) + 1), file_id_dict.keys())
        plt.savefig(os.path.join(path, "box_plot_per_cm_group_iter{}.png".format(self.current_train_iter)))
        plt.close(fig)

    def plot_uncertainty_boxplot_per_birads(self, output_folder=None):
        if output_folder is not None:
            path = output_folder
        else:
            path = os.path.join(self.output_folder, "test")
        b1_list = []
        b2_list = []
        b3_list = []
        b4_list = []
        b5_list = []
        with open(self.path_to_birads_labels, 'r') as f:
            reader = csv.reader(f)
            for (current_id, birads) in reader:
                try:
                    if int(birads) == 0:
                        b1_list.append(self.patient_stats.loc[current_id, 'uncertainty'])
                    elif int(birads) == 1:
                        b2_list.append(self.patient_stats.loc[current_id, 'uncertainty'])
                    elif int(birads) == 2:
                        b3_list.append(self.patient_stats.loc[current_id, 'uncertainty'])
                    elif int(birads) == 3:
                        b4_list.append(self.patient_stats.loc[current_id, 'uncertainty'])
                    elif int(birads) == 4:
                        b5_list.append(self.patient_stats.loc[current_id, 'uncertainty'])
                except KeyError:
                    continue

        file_id_dict = {'birads 1': b1_list,
                        'birads 2': b2_list,
                        'birads 3': b3_list,
                        'birads 4': b4_list,
                        'birads 5': b5_list}

        fig, ax = plt.subplots()
        ax.boxplot(file_id_dict.values())
        ax.set_title("Uncertainty per group")
        ax.set_ylim([0, 1])
        plt.xticks(range(1, len(file_id_dict.keys()) + 1), file_id_dict.keys())
        plt.savefig(os.path.join(path, "box_plot_per_birads_group_iter{}.png".format(self.current_train_iter)))
        plt.close(fig)

    def plot_metric_over_uncertainty(self, output_folder=None):
        if output_folder is not None:
            output_folder = os.path.join(output_folder,
                                         "uncertainty_plots", str(self.current_train_iter))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        else:
            output_folder = os.path.join(self.output_folder, "test",
                                         "uncertainty_plots", str(self.current_train_iter))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        uncertainty_thresh_list = [1., .9, .8, .7, .6, .5, .4, .3, .2]
        num_used_samples_list = []
        stats_per_uncertainty_thresh = pd.DataFrame(columns=list(self.used_metric_fns.keys()))
        for uncertainty_thresh in uncertainty_thresh_list:
            threshed_patients = self.patient_stats[self.patient_stats['uncertainty'] <= uncertainty_thresh]
            num_used_samples_list.append(len(threshed_patients.index))

            for metric_key, metric in self.used_metric_fns.items():
                try:
                    stats_per_uncertainty_thresh.loc[uncertainty_thresh, metric_key] = \
                        metric(threshed_patients.gt_label, threshed_patients.predicted_label,
                               threshed_patients.prediction, threshed_patients.predicted_probability)
                except IndexError:
                    # pandas series is empty due to no samples left below the current threshold
                    stats_per_uncertainty_thresh.loc[uncertainty_thresh, metric_key] = 0.0
                except Exception as e:
                    # pandas series is empty due to no samples left below the current threshold
                    stats_per_uncertainty_thresh.loc[uncertainty_thresh, metric_key] = 0.0

        for metric_key in self.used_metric_fns.keys():
            if metric_key != 'Confusion_Matrix':
                plt.figure()
                plt.plot(range(len(uncertainty_thresh_list)), stats_per_uncertainty_thresh[metric_key])
                plt.title(metric_key + ' over used samples')
                plt.xlabel('# used samples')
                plt.ylabel("value")
                plt.xticks(range(len(uncertainty_thresh_list)), num_used_samples_list)
                plt.savefig(os.path.join(output_folder, metric_key + ".png"))
                plt.close()

    def plot_roc_curve(self, gt_label=None, predicted_label=None, prediction=None, predicted_probability=None,
                       output_folder=None):
        if gt_label is None or predicted_probability is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_probabilities_list = np.array(list(self.patient_stats.predicted_probability))[:, 1]
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_probability,
                              pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_probabilities_list = np.array(list(predicted_probability))[:, 1]
        try:
            fpr, tpr, _ = roc_curve(gt_label_list, predicted_probabilities_list)
            roc_auc = auc(fpr, tpr)
            lw = 2
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area = {0})'.format(round(roc_auc, 2)))
            ax.plot([0, 1], color='navy', lw=lw)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            path = os.path.join(self.output_folder, 'test')
            if os.path.exists(path):
                file_name = os.path.join(self.output_folder, 'test',
                                         'roc_auc_iter{}.png'.format(self.current_train_iter))
            else:
                file_name = os.path.join(self.output_folder, 'roc_auc_extra.png')
            fig.savefig(file_name)
            plt.close(fig)
        except Exception as e:
            print(e)

    def plot_precision_recall_curve(self, gt_label=None, predicted_label=None, prediction=None,
                                    predicted_probability=None):
        if gt_label is None or predicted_probability is None:
            gt_label_list = list(self.patient_stats.gt_label)
            predicted_probabilities_list = np.array(list(self.patient_stats.predicted_probability))[:, 1]
        else:
            assert isinstance(gt_label, pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            assert isinstance(predicted_probability,
                              pd.Series), "Input to get_metric fns has to be instance of pandas.Series"
            gt_label_list = list(gt_label)
            predicted_probabilities_list = np.array(list(predicted_probability))[:, 1]
        try:
            lr_precision, lr_recall, _ = precision_recall_curve(gt_label_list, predicted_probabilities_list)

            no_skill = sum(gt_label_list) / len(gt_label_list)
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
            plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
            # axis labels
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve: '
                               'AP={0:0.2f}'.format(average_precision_score(gt_label_list,
                                                                            predicted_probabilities_list)))
            # show the legend
            plt.legend()
            path = os.path.join(self.output_folder, 'test')
            if os.path.exists(path):
                file_name = os.path.join(self.output_folder, 'test',
                                         'precision_recall_iter{}.png'.format(self.current_train_iter))
            else:
                file_name = os.path.join(self.output_folder, 'precision_recall_extra.png')
            plt.savefig(file_name)
            plt.close()
        except Exception as e:
            print(traceback.extract_stack())
            print(e)

    def print_metric_summary(self):
        for metric_name, metric_value in zip(list(self.train_cycle_stats.columns),
                                             list(self.train_cycle_stats.loc[self.current_train_iter])):
            print("{}: {}".format(metric_name, metric_value))

    # def print_metric_summaries(self, metric_summaries):
    #     for key, value in metric_summaries.items():
    #         self.print_metric_summary(key, value)

    # def save_metric_values(self, metric_key):
    #     self.train_cycle_stats[metric_key].to_csv(os.path.join(self.output_folder, metric_key + '.csv'))

    # def save_metric_summaries(self, metric_summaries):
    #     file_name = os.path.join(self.output_folder, 'summary.csv')
    #     utils.io.common.create_directories_for_file_name(file_name)
    #     with open(file_name, 'w') as file:
    #         writer = csv.writer(file)
    #         for key, value in metric_summaries.items():
    #             writer.writerow([key])
    #             writer.writerow(['mean'] + list(range(len(value) - 1)))
    #             writer.writerow(value)

    def write_softmax_predictions_per_id(self, output_folder=None):
        if output_folder is not None:
            path = output_folder
        else:
            path = os.path.join(self.output_folder, "test")
        csv_file = os.path.join(path, 'predicted_probability_iter{}.csv'.format(self.current_train_iter))

        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('Patient', 'Groundtruth', 'Pr(Benign (0))', 'Pr(Malignant (1))'))
            for current_id in self.patient_stats.index.values:
                writer.writerow(
                    (str(current_id),
                     int(self.patient_stats.loc[current_id, 'gt_label']),
                     float(self.patient_stats.loc[current_id, 'predicted_probability'][0]),
                     float(self.patient_stats.loc[current_id, 'predicted_probability'][1])))
            csvfile.close()

    def save_ids_to_tp(self, output_folder_cv=None):
        if output_folder_cv is not None:
            path = output_folder_cv
        else:
            path = os.path.join(self.output_folder, "test")
        tp_file_id_list = []
        tn_file_id_list = []
        fn_file_id_list = []
        fp_file_id_list = []
        for current_id in self.patient_stats.index.values:
            groundtruth_label = self.patient_stats.loc[current_id, 'gt_label']
            predicted_label = self.patient_stats.loc[current_id, 'predicted_label']
            if groundtruth_label == predicted_label:
                if groundtruth_label == 1:
                    tp_file_id_list.append(current_id)
                else:
                    tn_file_id_list.append(current_id)
            else:
                if groundtruth_label == 1:
                    fn_file_id_list.append(current_id)
                else:
                    fp_file_id_list.append(current_id)
        prediction_filepaths = {'tp': os.path.join(path, "tp.csv"),
                                'tn': os.path.join(path, "tn.csv"),
                                'fp': os.path.join(path, "fp.csv"),
                                'fn': os.path.join(path, "fn.csv")}
        file_id_dict = {'tp': tp_file_id_list, 'tn': tn_file_id_list, 'fp': fp_file_id_list, 'fn': fn_file_id_list}
        for metric in prediction_filepaths.keys():
            csv_file = prediction_filepaths[metric]
            list_to_write = file_id_dict[metric]
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([self.current_train_iter] + list_to_write)
                csvfile.close()

    def save_pandas_dataframe_to_csv(self, output_folder=None):
        if output_folder is not None:
            output_folder = output_folder
        else:
            output_folder = self.output_folder
        self.train_cycle_stats.to_csv(os.path.join(output_folder, 'test_summary.csv'))

    # ----------------- Finalize everything -----------

    def finalize(self):
        self.print_metric_summary()
        self.plot_softmax()
        if 'uncertainty' in self.loss_keyword:
            self.plot_uncertainty_boxplot_per_cm_group()
            self.plot_metric_over_uncertainty()
            self.plot_uncertainty_boxplot_per_birads()

        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        #self.plot_prediction_per_birads()
        self.write_softmax_predictions_per_id()
        self.save_ids_to_tp()
        self.save_pandas_dataframe_to_csv()

        # reset patient_stats
        if 'uncertainty' in self.loss_keyword:
            self.patient_stats = pd.DataFrame(
                columns=['gt_label', 'predicted_label', 'prediction', 'predicted_probability', 'uncertainty'])
        else:
            self.patient_stats = pd.DataFrame(
                columns=['gt_label', 'predicted_label', 'prediction', 'predicted_probability'])
        # self.save_metric_summaries(metric_summaries)

        # clean up
        plt.close('all')


class CrossValidationStatistics(MammoStatistics):
    def __init__(self,
                 classes,
                 current_iter,
                 output_folder,
                 loss_keyword,
                 metrics,
                 num_cv_folds):
        super().__init__(labels=classes,
                         current_train_iter=current_iter,
                         output_folder=output_folder,
                         loss_keyword=loss_keyword,
                         metrics=metrics)
        self.num_cv_folds = num_cv_folds
        self.fold_results = []
        self.full_cv_stats = pd.DataFrame(columns=list(self.used_metric_fns.keys()))
        if 'Confusion_Matrix' in list(self.used_metric_fns.keys()):
            self.full_cv_stats = self.full_cv_stats.drop(columns='Confusion_Matrix')

    def add_fold_results(self, train_cycle_stats, current_fold):
        try:
            train_cycle_stats_wocm = train_cycle_stats.drop('Confusion_Matrix')
        except KeyError:
            train_cycle_stats_wocm = train_cycle_stats.drop(columns='Confusion_Matrix')
        try:
            self.full_cv_stats.loc[current_fold] = train_cycle_stats_wocm
        except ValueError:
            self.full_cv_stats.loc[current_fold] = list(train_cycle_stats_wocm.iloc[0])

    def generate_latex_table(self):
        print(self.full_cv_stats.to_latex())
        with open(os.path.join(self.cv_output_folder, 'cv_stats_latex_table.txt'), 'w') as f:
            f.write(self.full_cv_stats.to_latex())
            f.close()

        cv_stats_rounded = self.full_cv_stats.round(decimals=3)
        with open(os.path.join(self.cv_output_folder, 'cv_stats_latex_table_rounded.txt'), 'w') as f:
            f.write(cv_stats_rounded.to_latex())
            f.close()

    def get_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.cv_output_folder = os.path.join(self.output_folder, 'cv_full')
        if not os.path.exists(self.cv_output_folder):
            os.makedirs(self.cv_output_folder)
        return self.output_folder

    def save_full_pandas_dataframe_to_csv(self):
        self.full_cv_stats.to_csv(os.path.join(self.output_folder, 'test_summary.csv'))

    def finalize(self):
        self.print_metric_summary()

        if 'uncertainty' in self.loss_keyword:
            self.plot_uncertainty_boxplot_per_cm_group(output_folder=self.cv_output_folder)
            self.plot_metric_over_uncertainty(output_folder=self.cv_output_folder)
            self.plot_uncertainty_boxplot_per_birads(output_folder=self.cv_output_folder)
        self.plot_prediction_per_birads(output_folder=self.cv_output_folder)
        self.plot_roc_curve(output_folder=self.cv_output_folder)
        self.plot_precision_recall_curve(output_folder=self.cv_output_folder)
        self.write_softmax_predictions_per_id(output_folder=self.cv_output_folder)
        self.save_ids_to_tp(output_folder_cv=self.cv_output_folder)
        self.save_pandas_dataframe_to_csv(output_folder=self.cv_output_folder)

        self.add_fold_results(self.train_cycle_stats, 'full')
        self.generate_latex_table()

        # clean up
        plt.close('all')
