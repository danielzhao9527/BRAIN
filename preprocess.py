import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

######################################################################
# random seed
# ~~~~~~~~~~~~~
SEED = 2030
np.random.seed(SEED)
tf.random.set_seed(SEED)

######################################################################
# Utils
# ~~~~~~~~~~~~~
# θ（4-8Hz）、α（8-13Hz）和β（13-30Hz）
theta_band = (4, 8)
alpha_band = (8, 12)
beta_band = (12, 30)
FS = 250


###################
# Augmentation
# ~~~~~~~~~~~~~
def augmentation_sr(alldata, label, batch_size=64):
    _, _, channels, timepoint = alldata.shape

    aug_data = []
    aug_label = []
    for cls4aug in range(4):
        cls_idx = np.where(label == cls4aug + 1)
        tmp_data = alldata[cls_idx]
        tmp_label = label[cls_idx]

        tmp_aug_data = np.zeros((int(batch_size / 4), 1, channels, timepoint))
        for ri in range(int(batch_size / 4)):
            for rj in range(9):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 9)
                tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:int(batch_size / 4)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]
    # combined with origin data
    aug_data = np.concatenate([alldata, aug_data])
    aug_label = np.concatenate([label, aug_label])
    return aug_data, aug_label


def load_data(data_path, subject, training, all_trials=True):
    n_channels = 22
    n_tests = 6 * 48
    window_Length = 7 * 250

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if (a_artifacts[trial] != 0 and not all_trials):
                continue
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_Length), :n_channels])
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]


def get_data(path, subject, isAugment=False):
    fs = 250  # sampling rate
    t1 = int(1.5 * fs)  # start at 1.5s
    t2 = int(6 * fs)  # end at 6s

    path = path + 's{:}/'.format(subject + 1)
    X_train, y_train = load_data(path, subject + 1, True)
    X_test, y_test = load_data(path, subject + 1, False)

    # prepare training data
    N_tr, N_ch, _ = X_train.shape
    X_train = X_train[:, :, t1:t2]

    if isAugment:  # Segment && Combine
        X_train = np.expand_dims(X_train, axis=1)
        X_train, y_train = augmentation_sr(X_train, y_train)
        X_train = np.squeeze(X_train, axis=1)

    X_train = np.expand_dims(X_train, axis=1)

    y_train_onehot = (y_train - 1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)

    N_test, N_ch, _ = X_test.shape
    X_test = X_test[:, :, t1:t2]
    X_test = np.expand_dims(X_test, axis=1)

    y_test_onehot = (y_test - 1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)

    # standardize
    for j in range(N_ch):
        scaler = StandardScaler()
        X_train[:, 0, j, :] = scaler.fit_transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot
