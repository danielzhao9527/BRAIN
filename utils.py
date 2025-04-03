import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import ConfusionMatrixDisplay


def draw_learning_curves(history, sub):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy of sub ' + str(sub))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss of sub ' + str(sub))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.close()


def draw_confusion_matrix(cf_matrix, sub, results_path):
    display_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                  display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix - Subject ' + sub)
    plt.savefig(results_path + '/subject_' + sub + '.png')
    plt.show()


def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub + 1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model ' + label + ' per subject')
    ax.set_ylim([0, 1])


def plt_tsne(data, label, n_sub, results_path, n_components=3):
    color_map = ['r', 'y', 'k', 'g', 'b', 'm', 'c']
    # The perplexity must be less than the number of samples, better choose [5, 50]
    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=30, n_iter=10000)
    X_tsne = tsne.fit_transform(data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], c=color_map[label[i]])

    plt.title("subject " + n_sub)
    # plt.show()
    plt.savefig(results_path + '/subject_' + n_sub + '.png', dpi=600)


def plt_topography(data, label):
    label = np.squeeze(np.transpose(label))
    idx = np.where(label == 1)
    data_draw = data[idx]

    mean_trial = np.mean(data_draw, axis=0)  # mean trial
    mean_trial = (mean_trial - np.mean(mean_trial)) / np.std(mean_trial)

    mean_ch = np.mean(mean_trial, axis=1)  # mean samples with channel dimension left

    # Draw topography
    my_montage = mne.channels.make_standard_montage('biosemi64')  # set a montage, see mne document
    index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56,
             29]
    my_montage.ch_names = [my_montage.ch_names[i] for i in index]
    my_montage.dig = [my_montage.dig[i + 3] for i in index]

    info = mne.create_info(ch_names=my_montage.ch_names, sfreq=250., ch_types='eeg')  # sample rate

    evoked1 = mne.EvokedArray(mean_trial, info)
    evoked1.set_montage(my_montage)

    plt.figure(1)
    im, cn = mne.viz.plot_topomap(mean_ch, evoked1.info, show=False)
    plt.colorbar(im)
    plt.show()
