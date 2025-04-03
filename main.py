import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import model as Model
from utils import *
from preprocess import get_data


def train(dataset_conf, train_conf, results_path):
    in_exp = time.time()
    best_models = open(results_path + "/best models.txt", "w")
    log_write = open(results_path + "/log.txt", "w")
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')

    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    is_augment = dataset_conf.get('isAugment')

    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    n_train = train_conf.get('n_train')

    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    # train all subjects
    for sub in range(n_sub):
        in_sub = time.time()
        print('\nTraining on subject ', sub + 1)
        log_write.write('\nTraining on subject ' + str(sub + 1) + '\n')
        # initiate variables
        BestSubjAcc = 0
        bestTrainingHistory = []
        # get training and test data
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(data_path, sub, isAugment=is_augment)

        for train in range(n_train):
            in_run = time.time()
            # create folders and files to save trained models for all runs
            filepath = results_path + '/saved models/run-{}'.format(train + 1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath = filepath + '/subject-{}.h5'.format(sub + 1)

            # create the model
            model = Model.DDF(n_classes=4, Chans=22, Samples=1125)
            # compile and train the model
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0,
                                save_best_only=True, save_weights_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
            model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train] = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)

            out_run = time.time()
            # print & write performance
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub + 1, train + 1,
                                                                           ((out_run - in_run) / 60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, train], kappa[sub, train])
            print(info)
            log_write.write(info + '\n')
            # always save better one
            if (BestSubjAcc < acc[sub, train]):
                BestSubjAcc = acc[sub, train]
                bestTrainingHistory = history

        # store best model
        best_run = np.argmax(acc[sub, :])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run + 1, sub + 1) + '\n'
        best_models.write(filepath)
        # print && write the best performance
        out_sub = time.time()
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub + 1, best_run + 1,
                                                                              ((out_sub - in_sub) / 60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]),
                                                                          acc[sub, :].std())
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run],
                                                                           np.average(kappa[sub, :]),
                                                                           kappa[sub, :].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info + '\n')
        # plot Learning curves
        print('Plot Learning Curves ....... ')
        draw_learning_curves(bestTrainingHistory, sub + 1)

    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format((out_exp - in_exp) / (60 * 60))
    print(info)
    log_write.write(info + '\n')

    np.savez(perf_allRuns, acc=acc, kappa=kappa)

    best_models.close()
    log_write.close()
    perf_allRuns.close()


def test(model, dataset_conf, results_path):
    log_write = open(results_path + "/log.txt", "a")
    best_models = open(results_path + "/best models.txt", "r")

    # get dataset paramters
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isAugment = dataset_conf.get('isAugment')

    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])

    # calculate the average performance
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
    perf_arrays = np.load(perf_allRuns)
    acc_allRuns = perf_arrays['acc']
    kappa_allRuns = perf_arrays['kappa']

    # model.summary()

    for sub in range(n_sub):
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, isAugment=isAugment)

        # load the best model weight
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])
        # predict
        tsne_data = model.predict(X_test)
        y_pred = tsne_data.argmax(axis=-1)
        labels = y_test_onehot.argmax(axis=-1)
        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)

        # draw confusion matrix
        plt.rcParams.update({'font.size': 12})
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='pred')
        draw_confusion_matrix(cf_matrix[sub, :, :], str(sub + 1), results_path)

        # print && write performance
        info = 'Subject: {}   best_run: {:2}  '.format(sub + 1,
                                                       (filepath[filepath.find('run-') + 4:filepath.find('/sub')]))
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_bestRun[sub], kappa_bestRun[sub])
        info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub, :].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub, :].std())
        print(info)
        log_write.write('\n' + info)

    # print & write the average performance
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun))
    info = info + '\nAverage of {} subjects x {} runs (average of {} experiments):\nAccuracy = {:.4f}   Kappa = {:.4f}'.format(
            n_sub, acc_allRuns.shape[1], (n_sub * acc_allRuns.shape[1]),
            np.average(acc_allRuns), np.average(kappa_allRuns))
    print(info)
    log_write.write(info)

    draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy')
    draw_performance_barChart(n_sub, kappa_bestRun, 'K-score')
    draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path)
    log_write.close()


def run(isTrain = True):
    data_path = "./data/"
    results_path = "./results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # dataset paramters
    dataset_conf = {'n_classes': 4, 'n_sub': 9, 'n_channels': 22, 'data_path': data_path, 'isAugment': True}
    # training hyperparamters
    train_conf = {'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.0009, 'n_train': 5}

    if isTrain:
        train(dataset_conf, train_conf, results_path)
    else:
        model = Model.DDF(n_classes=4, Chans=22, Samples=1125)
        test(model, dataset_conf, results_path)


# use last gpu as default
def set_GPU(which_gpu = -1):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("-----------already set memory growth------------")
    tf.config.experimental.set_visible_devices(gpus[which_gpu], 'GPU')


if __name__ == "__main__":
    set_GPU()
    run(isTrain=True)
