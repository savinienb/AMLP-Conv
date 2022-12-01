from textwrap import wrap
import re
import itertools
import tfplot
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import FixedLocator, FixedFormatter


def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix',
                          tensor_name = 'MyFigure/image', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    # fig = matplotlib.figure.Figure(figsize=(4, 4), dpi=320, facecolor='w', edgecolor='k')
    # ax = fig.add_subplot(1, 1, 1)
    fig, ax = matplotlib.pyplot.subplots(figsize=(4, 4), dpi=320, facecolor='w', edgecolor='k')
    im = ax.imshow(cm, cmap='Oranges')

    #labels = [str(x + 1) for x in labels]
    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes)).astype(np.float32)
    # current_xticks = ax.get_xticks()
    # current_yticks = ax.get_yticks()
    x_formatter = FixedFormatter(classes)
    y_formatter = FixedFormatter(classes)
    x_locator = FixedLocator(tick_marks)
    y_locator = FixedLocator(tick_marks)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)

    ax.set_xlabel('Predicted', fontsize=7)
    #ax.set_xticks(tick_marks)
    #c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    #ax.set_xticklabels((np.int32(ax.get_xticks()) + 1).astype(str), fontsize=4, rotation=-90,  ha='center')
    ax.tick_params(axis="x", labelsize=4) #, labelrotation=-90)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    #ax.set_yticks(tick_marks)
    #ax.set_yticklabels(classes, fontsize=4, va ='center')
    #ax.set_yticklabels((np.int32(ax.get_yticks()) + 1).astype(str), fontsize=4, va='center')
    ax.tick_params(axis="y", labelsize=4)
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(False)
    matplotlib.pyplot.show()
    summary = '' #tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


if __name__ == '__main__':
    #plot_confusion_matrix(np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]), np.array([0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0]), ['bob', 'man'])

    cm = np.array([[83, 77], [28, 132]])
    labels = ['BI-RADS>3', 'BI-RADS<=3']
    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    # fig = matplotlib.figure.Figure(figsize=(4, 4), dpi=320, facecolor='w', edgecolor='k')
    # ax = fig.add_subplot(1, 1, 1)
    fig, ax = matplotlib.pyplot.subplots(figsize=(4, 4), dpi=320, facecolor='w', edgecolor='k')
    im = ax.imshow(cm, cmap='Oranges')

    # labels = [str(x + 1) for x in labels]
    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes)).astype(np.float32)
    # current_xticks = ax.get_xticks()
    # current_yticks = ax.get_yticks()
    x_formatter = FixedFormatter(classes)
    y_formatter = FixedFormatter(classes)
    x_locator = FixedLocator(tick_marks)
    y_locator = FixedLocator(tick_marks)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)

    ax.set_xlabel('Predicted', fontsize=7)
    # ax.set_xticks(tick_marks)
    # c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    # ax.set_xticklabels((np.int32(ax.get_xticks()) + 1).astype(str), fontsize=4, rotation=-90,  ha='center')
    ax.tick_params(axis="x", labelsize=4)  # , labelrotation=-90)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    # ax.set_yticks(tick_marks)
    # ax.set_yticklabels(classes, fontsize=4, va ='center')
    # ax.set_yticklabels((np.int32(ax.get_yticks()) + 1).astype(str), fontsize=4, va='center')
    ax.tick_params(axis="y", labelsize=4)
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="black")
    fig.set_tight_layout(False)
    matplotlib.pyplot.show()
