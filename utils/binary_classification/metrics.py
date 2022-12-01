
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.io.image import write_np
import skimage
import scipy
from sklearn.metrics import roc_curve, auc, f1_score, balanced_accuracy_score
from scipy.ndimage import zoom
import utils
# class MetricBase(object):
#     def __call__(self,  predicted_label, groundtruth_label):
#         raise NotImplementedError()

class PatientMetrics(object):
    def __init__(self,  current_id, predicted_label, groundtruth_label, prediction):
        self.predicted_label = predicted_label
        self.groundtruth_label = groundtruth_label
        self.prediction = prediction
        self.current_id = current_id

        self.non_binary_prediction = self.get_non_binary_prediction()

    def get_prediction_list(self):
        return self.current_id, self.predicted_label, self.groundtruth_label

    def calculate_statistics(self):
        tp = self.groundtruth_label == 1 and self.groundtruth_label == self.predicted_label
        tn = self.groundtruth_label == 0 and self.groundtruth_label == self.predicted_label
        fp = self.groundtruth_label == 0 and self.predicted_label == 1
        fn = self.groundtruth_label == 1 and self.predicted_label == 0

        return tp, tn, fp, fn

    def get_non_binary_prediction(self):
        non_binary_prediction = scipy.special.softmax(self.prediction, axis=0)
        return non_binary_prediction


   # def __call__(self,  predicted_label, groundtruth_label):
   #     return self.calculate_statistics(predicted_label, groundtruth_label)



class CohortMetrics(object):
    """
    The dataset that describes the entire image data set.
    """
    def __init__(self, metric_summary, output_folder, current_iter, max_iter):
        """
             Initializer.
             :param metric_summary: dictionary containing list of tp, tn, fp, fn (bool) for each patient
        """
        self.output_folder = output_folder
        self.num_tp = int(np.sum(metric_summary['tp']))
        self.num_tn = int(np.sum(metric_summary['tn']))
        self.num_fp = int(np.sum(metric_summary['fp']))
        self.num_fn = int(np.sum(metric_summary['fn']))

        self.acc = self.get_accuracy()
        self.sensitivity = self.get_sensitivity()
        self.specificity = self.get_specificity()
        self.current_iter = current_iter

        if current_iter == max_iter:
            self.groundtruth = metric_summary['groundtruth']
            self.softmax = list((metric_summary['groundtruth'], metric_summary['softmax']))
            self.plot_softmax()
            self.roc_curve()


    def get_accuracy(self):
        correct = np.sum((self.num_tp, self.num_tn))
        total = np.sum((self.num_tp, self.num_tn, self.num_fp, self.num_fn))

        return np.divide(correct, total)

    def get_sensitivity(self):
        return self.num_tp / (self.num_tp + self.num_fn) if self.num_tp + self.num_fn > 0 else 1


    def get_specificity(self):
        return self.num_tn / (self.num_tn + self.num_fp) if self.num_tn + self.num_fp > 0 else 1

    def plot_softmax(self):
        softmax_pos = []
        softmax_neg = []
        for patient in range(len(self.softmax[0])):
            if self.softmax[0][patient] == 1:
                softmax_pos.append(self.softmax[1][patient])
            else:
                softmax_neg.append(self.softmax[1][patient])
        fig = plt.figure()
        plt.hist(softmax_pos, bins=10, color='red', label='Malignant')
        plt.hist(softmax_neg, bins=10, color='green', alpha=0.4, label='Benign')
        plt.legend()
        plt.title('Malignant (positive) prediction probabilities')
        file_name = os.path.join(self.output_folder, 'test','softmax_histogram_iter{}.png'.format(self.current_iter))
        print(file_name)
        fig.savefig(file_name)
        plt.close()

    def roc_curve(self):
        # for malignant class
        fpr, tpr, _ = roc_curve(self.groundtruth, self.softmax[1])
        roc_auc = auc(fpr, tpr)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area = {0})'.format(round(roc_auc,2)))
        plt.plot([0, 1], color='navy', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        file_name = os.path.join(self.output_folder, 'test', 'roc_auc_iter{}.png'.format(self.current_iter))
        plt.savefig(file_name)


class NetworkMetrics(object):
    """
        Provides a class for network metrics such as convolution filters, intermediate activations etc.
    """
    def __init__(self, intermediate_activation, output_folder, current_id, generators, current_iter, prediction, true_label):
        """
             Initializer.
             :param metric_summary: dictionary containing list of tp, tn, fp, fn (bool) for each patient
        """
        self.true_label = true_label
        self.prediction = prediction
        self.predicted_label = np.argmax(prediction)
        self.outputs = intermediate_activation
        self.current_id = current_id
        self.current_iter = current_iter
        self.generators = generators
        self.image_size = self.generators['image_stack_L'].shape
        self.output_folder = output_folder
        self.activation_output_folder = os.path.join(self.output_folder, 'activations/', self.current_id[:-8])
        #self.image = self.plot_summed_intermediate_activations(select_channel=0, select_feature=1)#self.plot_intermediate_activations(select_channel=0, select_feature=1)
        #self.image_mip = self.plot_mip_intermediate_activations(0, 1)
        #self.plot_mip_summed_intermediate_activations(0, 1)
        self.plot_cam_activations(0, 1)
        #self.save_intermediate_activation_images()
        #self.heatmap = self.calculate_heatmaps()
        #self.plot_heatmaps()

    def plot_current_image(self):
        plt.imshow(np.rot90(self.generators['image_stack_L'][1, int(self.image_size[1] / 2), :, :], 3), cmap='jet')
        plt.colorbar(ticks=ticker.MaxNLocator(nbins=4))
        plt.title('Patient: {0} (L)'.format(self.current_id))
        file_name = os.path.join(self.output_folder, self.current_id[:-8], 'L_image_stack.png')
        plt.savefig(file_name)
        plt.close()

    def plot_intermediate_activations(self, select_channel, select_feature):
        if not os.path.exists(self.activation_output_folder):
            os.makedirs(self.activation_output_folder)

        plot_max = {'net_activations': [0, 2],
                    'net_activations_L': [0, 2],
                    'net_activations_R': [0, 2]}

        channel = 0 # corresponds to the channels in the image stack

        for net_key in self.outputs.keys():
            if 'activation' in net_key:
                act_img_dict = self.outputs[net_key]
                for node_key in act_img_dict.keys():
                    activation_image = act_img_dict[node_key][channel, :, :, :, :]
                    sub_y = 8
                    sub_x = int(activation_image.shape[0]/sub_y)
                    fig, ax = plt.subplots(sub_x, sub_y)
                    fig.subplots_adjust(hspace=0, wspace=0)
                    ax = ax.ravel()
                    for ft in range(0, len(ax)):
                        single_feature_plt = ax[ft].imshow(np.rot90(activation_image[ft, int(activation_image.shape[1]/2), :, :], 3),
                                                           cmap='jet', vmin=plot_max[net_key][0], vmax=plot_max[net_key][1])
                        ax[ft].axes.get_yaxis().set_visible(False)
                        ax[ft].axes.get_xaxis().set_visible(False)

                    fig.colorbar(single_feature_plt, ax=fig.get_axes(), ticks=ticker.MaxNLocator(nbins=4))
                    fig.suptitle('{0}_{1}'.format(net_key, node_key), fontsize=12)

                    file_name = os.path.join(self.activation_output_folder, 'iter_{0}_{1}_{2}.png'.format(self.current_iter, net_key, node_key))
                    fig.savefig(file_name)
                    plt.close()

            image = act_img_dict['conv1_1'][select_channel, select_feature, :, :, :]
        return image

    def plot_summed_intermediate_activations(self, select_channel, select_feature):
        if not os.path.exists(self.activation_output_folder):
            os.makedirs(self.activation_output_folder)

        channel = 0 # corresponds to the channels in the image stack

        for net_key in self.outputs.keys():
            if 'activation' in net_key:
                act_img_dict = self.outputs[net_key]
                fig, ax = plt.subplots(1, len(act_img_dict.keys()), figsize=[10.24,10.24])
                if len(act_img_dict.keys()) > 1:
                    ax = ax.ravel()
                cnt = 0
                for node_key in act_img_dict.keys():
                    activation_image = act_img_dict[node_key][channel, :, :, :, :]
                    activation = activation_image[:, int(activation_image.shape[1]/2), :, :]
                    if len(act_img_dict.keys()) > 1:
                        image = np.rot90(np.sum(activation, 0), 4)
                        single_node = ax[cnt].imshow(image, cmap='jet')
                        ax[cnt].axes.get_yaxis().set_visible(False)
                        ax[cnt].axes.get_xaxis().set_visible(False)
                        ax[cnt].title.set_text(str(node_key)[:-2])
                    else:
                        image = np.rot90(np.sum(activation, 0), 4)
                        single_node = ax.imshow(image, cmap='jet')
                        ax.axes.get_yaxis().set_visible(False)
                        ax.axes.get_xaxis().set_visible(False)
                        ax.title.set_text(str(node_key)[:-2])
                    #fig.colorbar(single_node, ticks=ticker.MaxNLocator(nbins=4))
                    fig.suptitle(net_key, fontsize=12)
                    cnt = cnt + 1
                    file_name = os.path.join(self.activation_output_folder, 'activation_summed_{0}_{1}.png'.format(net_key, node_key))
                fig.savefig(file_name)
                plt.close()
        return image

    def plot_mip_summed_intermediate_activations(self, select_channel, select_feature):
        if not os.path.exists(self.activation_output_folder):
            os.makedirs(self.activation_output_folder)

        channel = 0 # corresponds to the channels in the image stack

        for net_key in self.outputs.keys():
            if 'activation' in net_key:
                act_img_dict = self.outputs[net_key]
                fig, ax = plt.subplots(1, len(act_img_dict.keys()), figsize=[10.24,10.24])
                if len(act_img_dict.keys()) > 1:
                    ax = ax.ravel()
                cnt = 0
                for node_key in act_img_dict.keys():
                    activation_image = act_img_dict[node_key][channel, :, :, :, :]
                    image = np.rot90(np.sum(np.max(activation_image, 1), 0), 4)
                    if len(act_img_dict.keys()) > 1:
                        single_node = ax[cnt].imshow(image, cmap='jet')
                        ax[cnt].axes.get_yaxis().set_visible(False)
                        ax[cnt].axes.get_xaxis().set_visible(False)
                        ax[cnt].title.set_text(str(node_key)[:-2])
                    else:
                        single_node = ax.imshow(image, cmap='jet')
                        ax.axes.get_yaxis().set_visible(False)
                        ax.axes.get_xaxis().set_visible(False)
                        ax.title.set_text(str(node_key)[:-2])
                    #fig.colorbar(single_node, ticks=ticker.MaxNLocator(nbins=4))
                    fig.suptitle(net_key, fontsize=12)
                    cnt = cnt + 1
                    file_name = os.path.join(self.activation_output_folder, 'activation_mip_summed_{0}_{1}.png'.format(net_key, node_key))
                fig.savefig(file_name)
                plt.close()
        return image

    def plot_cam_activations(self, select_channel, select_feature):
        if not os.path.exists(self.activation_output_folder):
            os.makedirs(self.activation_output_folder)

        if self.current_id.startswith('SKYRA_MMAM_1010'):
            pass
        else:
            return 1, 1

        channel = 0 # corresponds to the channels in the image stack
        act_array = np.squeeze(np.concatenate((self.outputs['net_activations_L']['reluLast'],
                                    self.outputs['net_activations_R']['reluLast']),
                                   axis=1))
        act_left = self.outputs['net_activations_L']['reluLast']
        act_right = self.outputs['net_activations_R']['reluLast']
        weight_array = self.outputs['net_weights']['prediction']
        zoom_factor = 16.0
        # cam_low_res_L = np.zeros((act_array.shape[1], act_array.shape[2], act_array.shape[3]))
        # cam_low_res_R = np.zeros((act_array.shape[1], act_array.shape[2], act_array.shape[3]))
        # num_weights_per_breast_side = int(weight_array.shape[0] / 2)
        for class_index in range(weight_array.shape[1]):
            # for i in range(num_weights_per_breast_side):
            #     cam_low_res_L += weight_array[i, class_index] * act_array[i, :, :, :]
            #     cam_low_res_R += weight_array[i + num_weights_per_breast_side, class_index] * \
            #                      act_array[i + num_weights_per_breast_side, :, :, :]

            cam_low_res_L = np.squeeze(np.dot(weight_array[:128, class_index][np.newaxis, np.newaxis, np.newaxis, :],
                                      np.transpose(act_array[:128], (1, 2, 0, 3))))
            cam_low_res_R = np.squeeze(np.dot(weight_array[128:, class_index][np.newaxis, np.newaxis, np.newaxis, :],
                                      np.transpose(act_array[128:], (1, 2, 0, 3))))
            # print("weird calculation working: {}, {}".format(np.allclose(cam_low_res_L, cam_low_res_L1),
            #                                                  np.allclose(cam_low_res_R, cam_low_res_R1)))
            cam_orig_res_L = zoom(np.squeeze(cam_low_res_L), zoom=zoom_factor, mode='constant')
            cam_orig_res_R = zoom(np.squeeze(cam_low_res_R), zoom=zoom_factor, mode='constant')
            #cam_orig_res_norm = 2.*(cam_orig_res - np.min(cam_orig_res))/np.ptp(cam_orig_res)-1
            cam_orig_res_standardized_L = (cam_orig_res_L - np.mean(cam_orig_res_L)) / np.std(cam_orig_res_L)
            cam_orig_res_standardized_R = (cam_orig_res_R - np.mean(cam_orig_res_R)) / np.std(cam_orig_res_R)
            plot_single_slices = True
            if plot_single_slices:
                fig, ax = plt.subplots(1, 2, figsize=[10.24, 20.48])
                # ax_init_pos_0 = ax[0]._originalPosition
                # ax_init_pos_1 = ax[1]._originalPosition
                for i in range(cam_orig_res_standardized_L.shape[1]):
                    for j in range(len(ax)):
                        ax[j].axes.get_yaxis().set_visible(False)
                        ax[j].axes.get_xaxis().set_visible(False)
                    ax[0].title.set_text(str('CAM_R'))
                    ax[1].title.set_text(str('CAM_L'))
                    fig.suptitle('CAM', fontsize=12)
                    single_node3 = ax[0].imshow(cam_orig_res_standardized_R[i, :, :], cmap='jet')
                    single_node3_1 = ax[1].imshow(cam_orig_res_standardized_L[i, :, :], cmap='jet')
                    cbar = fig.colorbar(single_node3, ax=ax.ravel(), shrink=0.8)
                    file_name = os.path.join(self.activation_output_folder, 'CAM_c{}_slice{}.png'.format(class_index, i))
                    fig.savefig(file_name)
                    cbar.remove()
                    fig.subplots_adjust(right=0.9)
                    #plt.show()
                plt.close()

            fname = os.path.join(self.activation_output_folder, 'CAM_L_c{}_predictedLabel{}_trueLabel{}.mha'.format(
                class_index, self.predicted_label, self.true_label))
            utils.io.image.write_np(cam_orig_res_standardized_L, fname)
            fname = os.path.join(self.activation_output_folder, 'CAM_R_c{}_predictedLabel{}_trueLabel{}.mha'.format(
                class_index, self.predicted_label, self.true_label))
            utils.io.image.write_np(cam_orig_res_standardized_R, fname)
        debug_im_fname_L = os.path.join(self.activation_output_folder, 'debug_im_L.mha')
        utils.io.image.write_np(self.generators['image_stack_L'][0, :, :, :], debug_im_fname_L)
        debug_im_fname_R = os.path.join(self.activation_output_folder, 'debug_im_R.mha')
        utils.io.image.write_np(self.generators['image_stack_R'][0, :, :, :], debug_im_fname_R)
        return cam_orig_res_standardized_R, cam_orig_res_standardized_L

    def plot_mip_intermediate_activations(self, select_channel, select_feature):
        if not os.path.exists(self.activation_output_folder):
            os.makedirs(self.activation_output_folder)

        #channel = 0 # corresponds to the channels in the image stack

        for net_key in self.outputs.keys():
            if 'activation' in net_key:
                act_img_dict = self.outputs[net_key]
                for node_key in act_img_dict.keys():
                    fig, ax = plt.subplots(7, 6, figsize=[10.24, 10.24])
                    ax = ax.ravel()
                    activation_image = act_img_dict[node_key][0, :, :, :, :]
                    for channel in range(ax.shape[0] if activation_image.shape[0] > 7*6 else activation_image.shape[0]):
                        activation = activation_image[channel, :, :, :]
                        image = np.rot90(np.max(activation, 1), 4)
                        single_node = ax[channel].imshow(image, cmap='jet')
                        ax[channel].axes.get_yaxis().set_visible(False)
                        ax[channel].axes.get_xaxis().set_visible(False)
                        #ax[channel].title.set_text(str(node_key)[:-2])
                        #fig.colorbar(single_node, ticks=ticker.MaxNLocator(nbins=4))
                    fig.suptitle(net_key + str(node_key), fontsize=12)
                    file_name = os.path.join(self.activation_output_folder, 'activation_mip_{0}_{1}.png'.format(net_key, node_key))
                    fig.savefig(file_name)
                    plt.close()
        return image

    def save_intermediate_activation_images(self):
        write_np(self.image, os.path.join(self.activation_output_folder, "iter_{0}{1}.nii.gz".format(str(self.current_iter), self.current_id.replace("/", "_") )))

    def calculate_heatmaps(self):

        ##TODO incorrect implementation, WIP, needs attention
        # Get validation heat map

        weights = self.outputs['net_weights']['conv4_1'][0, 0, 0, 0, :]
        prediction_activations = self.outputs['net_activations_R']['conv4_1']

        heatmap = np.matmul(weights.reshape(-1, 64), prediction_activations.transpose())
        heatmap = np.reshape(heatmap, [8, 8, 8])

        return heatmap

    def plot_heatmaps(self):
        heatmap128 = skimage.transform.resize(self.heatmap, [128, 128, 128])

        fig, ax = plt.subplots(2, 2)
        ax.ravel()
        ax[0].imshow(self.heatmap[4, :, :], cmap='jet')
        ax[1].imshow(heatmap128[int(self.image_size[0] / 2)+4, :, :], cmap='jet')
        ax[2].imshow(self.generators['image_stack_R'][0, int(self.image_size[0] / 2)+4, :, :], cmap='gray')
        ax[3].imshow(self.generators['image_stack_R'][0, int(self.image_size[0] / 2)+4, :, :], cmap='gray')
        ax[3].imshow(heatmap128[int(self.image_size[0] / 2), :, :], cmap='jet', alpha=0.2)
        #fig.axes.get_xaxis().set_visible(False)
        #fig.axes.get_yaxis().set_visible(False)
        plt.show()
