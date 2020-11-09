import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import trange
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Masking, Bidirectional, LSTM, Dense
from tensorflow.keras.losses import KLDivergence, MSE
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean, Accuracy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dualstudent.metrics import PhoneErrorRate
from dualstudent.models._utils import sigmoid_rampup, select_batch
from dualstudent.preprocess import map_labels


class DualStudent(Model):
    """"
    Dual Student for Automatic Speech Recognition (ASR).

    How to train: 1) set the optimizer by means of compile(), 2) use train()
    How to test: use test()
    How to predict: use pad_and_predict()

    Remarks:
    - Do not use fit() by Keras, use train()
    - Do not use evaluate() by Keras, use test()
    - Do not use predict() by Keras, use pad_and_predict()
    - Compiled metrics and loss (i.e. set by means of compile()) are not used

    Original proposal for image classification: https://arxiv.org/abs/1909.01804
    """

    def __init__(self, n_classes, n_hidden_layers=3, n_units=96, consistency_loss='mse', consistency_scale=10,
                 stabilization_scale=100, epsilon=0.6, padding_value=0., sigma=0.01, version='mono_directional'):
        """
        Constructs a Dual Student model.

        :param n_classes: number of classes (i.e. number of units in the last layer of each student)
        :param n_hidden_layers: number of hidden layers in each student (i.e. LSTM layers)
        :param n_units: number of units for each hidden layer
        :param consistency_loss: one of 'mse', 'kl'
        :param consistency_scale: maximum value of weight for consistency constraint
        :param stabilization_scale: maximum value of weight for stabilization constraint
        :param epsilon: threshold for stable sample
        :param padding_value: value used to pad input sequences (used as mask_value for Masking layer)
        :param sigma: standard deviation for noisy augmentation
        :param version: one of:
            - 'mono_directional': both students have mono-directional LSTM layers
            - 'bidirectional: both students have bidirectional LSTM layers
            - 'imbalanced': one student has mono-directional LSTM layers, the other one bidirectional
        """
        super(DualStudent, self).__init__()

        # store parameters
        self.n_classes = n_classes
        self.padding_value = padding_value
        self.n_units = n_units
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.consistency_scale = consistency_scale
        self.stabilization_scale = stabilization_scale
        self.sigma = sigma
        self.version = version
        self._lambda1 = None
        self._lambda2 = None

        # loss (weighted sum of classification, consistency and stabilization losses)
        self._loss1 = Mean(name='loss1')
        self._loss2 = Mean(name='loss2')

        # classification loss
        self._loss_cls = SparseCategoricalCrossentropy()        # to be used on-the-fly
        self._loss1_cls = Mean(name='loss1_cls')                # for history on the epoch
        self._loss2_cls = Mean(name='loss2_cls')

        # consistency loss
        if consistency_loss == 'mse':
            self._loss_con = MeanSquaredError()                 # to be used on-the-fly
        elif consistency_loss == 'kl':
            self._loss_con = KLDivergence()
        else:
            raise ValueError('Invalid consistency metric')
        self._loss1_con = Mean(name='loss1_con')                # for history on the epoch
        self._loss2_con = Mean(name='loss2_con')

        # stabilization loss
        self._loss_sta = MeanSquaredError()                     # to be used on-the-fly
        self._loss1_sta = Mean(name='loss1_sta')                # for history on the epoch
        self._loss2_sta = Mean(name='loss2_sta')

        # accuracy
        self._acc1 = SparseCategoricalAccuracy(name='acc1')
        self._acc2 = SparseCategoricalAccuracy(name='acc2')

        # compose students
        if version == 'mono_directional':
            lstm_types = ['mono_directional', 'mono_directional']
        elif version == 'bidirectional':
            lstm_types = ['bidirectional', 'bidirectional']
        elif version == 'imbalanced':
            lstm_types = ['mono_directional', 'bidirectional']
        else:
            raise ValueError('Invalid student version')
        self.student1 = self._get_student('student1', lstm_types[0])
        self.student2 = self._get_student('student2', lstm_types[1])

        # masking layer (just to use compute_mask)
        self.mask = Masking(mask_value=self.padding_value)

    def _get_student(self, name, lstm_type):
        student = Sequential(name=name)
        student.add(Masking(mask_value=self.padding_value))
        if lstm_type == 'mono_directional':
            for i in range(self.n_hidden_layers):
                student.add(LSTM(units=self.n_units, return_sequences=True))
        elif lstm_type == 'bidirectional':
            for i in range(self.n_hidden_layers):
                student.add(Bidirectional(LSTM(units=self.n_units, return_sequences=True)))
        else:
            raise ValueError('Invalid LSTM version')
        student.add(Dense(units=self.n_classes, activation="softmax"))
        return student

    def _noisy_augment(self, x):
        return x + tf.random.normal(shape=x.shape, stddev=self.sigma)

    def call(self, inputs, training=False, student='student1', **kwargs):
        """
        Feed-forwards inputs to one of the students.

        This function is called internally by __call__(). Do not use it directly, use the model as callable. You may
        prefer to use pad_and_predict() instead of this, because it pads the sequences and splits in batches. For a big
        dataset, it is strongly suggested that you use pad_and_predict().

        :param inputs: tensor of shape (batch_size, n_frames, n_features)
        :param training: boolean, whether the call is in inference mode or training mode
        :param student: one of 'student1', 'student2'
        :return: tensor of shape (batch_size, n_frames, n_classes), softmax activations (probabilities)
        """
        if student == 'student1':
            return self.student1(inputs, training=training)
        elif student != 'student1':
            return self.student2(inputs, training=training)
        else:
            raise ValueError('Invalid student')

    def pad_and_predict(self, x, student='student1', batch_size=32):
        """
        It pads the inputs and feed-forwards to one of the students, one batch at a time. You should use this method
        if you wish to pass to the model variable-length utterances or a big dataset.

        :param x: numpy array of numpy arrays (n_frames, n_features), features corresponding to y_labeled.
            'n_frames' can vary, padding is added to make x_labeled a tensor.
        :param student: one of 'student1', 'student2'
        :param batch_size: batch size
        :return: tuple (predictions, mask), where:
            - predictions is numpy array of shape (n_utterances, n_frames, n_classes) containing the softmax activation
                of the padded inputs.
            - mask is the boolean mask having True for the non-padding time steps.
        """
        y_pred = None
        n_batches = int(len(x) / batch_size)
        x = pad_sequences(x, padding='post', value=self.padding_value, dtype='float32')
        mask = self.mask.compute_mask(x)

        for i in range(n_batches):
            x_batch = select_batch(x, i, batch_size)
            x_batch = tf.convert_to_tensor(x_batch)
            y_batch = self(x_batch, student=student, training=False)
            y_pred = tf.concat([y_pred, y_batch], axis=0) if y_pred is not None else y_batch

        return y_pred.numpy(), mask

    def train(self, x_labeled, x_unlabeled, y_labeled, x_val=None, y_val=None, n_epochs=10, batch_size=32, shuffle=True,
              evaluation_mapping=None, logs_path=None, checkpoints_path=None, seed=None):
        """
        Trains the model with both labeled and unlabeled data (semi-supervised learning).

        :param x_labeled: numpy array of numpy arrays (n_frames, n_features), features corresponding to y_labeled.
            'n_frames' can vary, padding is added to make 'x_labeled' a tensor.
        :param x_unlabeled: numpy array of numpy arrays of shape (n_frames, n_features), features without labels.
            'n_frames' can vary, padding is added to make 'x_unlabeled' a tensor.
        :param y_labeled: numpy array of numpy arrays of shape (n_frames,), labels corresponding to x_labeled.
            'n_frames' can vary, padding is added to make 'y_labeled' a tensor.
        :param x_val: like x_labeled, but for validation set
        :param y_val: like y_labeled, but for validation set
        :param n_epochs: integer, number of training epochs
        :param batch_size: integer, batch size
        :param shuffle: boolean, whether to shuffle at each epoch or not
        :param evaluation_mapping: dictionary {training label -> test label}, the test phones should be a subset of the
            training phones
        :param logs_path: path where to save logs for TensorBoard
        :param checkpoints_path: path to a file or a directory:
            - In case of a file, the checkpoint at that path will be restored and the training will continue from there.
              New checkpoints will be saved in the parent directory.
            - In case of a directory, the training will start from scratch and the checkpoints will be saved
              at that path.
        :param seed: seed for the random number generator
        """
        # set seed
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # setup for logs and checkpoints
        train_summary_writer = tf.summary.create_file_writer(logs_path)
        checkpoint = None
        if checkpoints_path is not None:
            checkpoints_path = Path(checkpoints_path)
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

            # if file, restore the checkpoint and set path for further checkpoints to parent
            if checkpoints_path.is_file():
                self.build(input_shape=(None,) + x_labeled.shape[1:])
                checkpoint.restore(tf.train.latest_checkpoint(str(checkpoints_path)))
                checkpoints_path = checkpoints_path.parent
            checkpoints_path = str(checkpoints_path)

        # compute batch sizes
        labeled_batch_size = int(len(x_labeled) / (len(x_unlabeled) + len(x_labeled)) * batch_size)
        unlabeled_batch_size = batch_size - labeled_batch_size
        n_batches = min(int(len(x_unlabeled) / unlabeled_batch_size), int(len(x_labeled) / labeled_batch_size))

        # training loop
        for epoch in trange(n_epochs, desc='epochs'):
            # ramp up lambda1 and lambda2
            self._lambda1 = self.consistency_scale * sigmoid_rampup(epoch, rampup_length=5)
            self._lambda2 = self.stabilization_scale * sigmoid_rampup(epoch, rampup_length=5)

            # shuffle training set
            if shuffle:
                indices = np.arange(len(x_labeled))  # get indices to shuffle coherently features and labels
                np.random.shuffle(indices)
                x_labeled = x_labeled[indices]
                y_labeled = y_labeled[indices]
                np.random.shuffle(x_unlabeled)

            for i in trange(n_batches, desc='batches'):
                # select batch
                x_labeled_batch = select_batch(x_labeled, i, labeled_batch_size)
                x_unlabeled_batch = select_batch(x_unlabeled, i, unlabeled_batch_size)
                y_labeled_batch = select_batch(y_labeled, i, labeled_batch_size)

                # pad batch
                x_labeled_batch = pad_sequences(x_labeled_batch, padding='post', value=self.padding_value,
                                                dtype='float32')
                x_unlabeled_batch = pad_sequences(x_unlabeled_batch, padding='post', value=self.padding_value,
                                                  dtype='float32')
                y_labeled_batch = pad_sequences(y_labeled_batch, padding='post', value=-1)

                # convert to tensors
                x_labeled_batch = tf.convert_to_tensor(x_labeled_batch)
                x_unlabeled_batch = tf.convert_to_tensor(x_unlabeled_batch)
                y_labeled_batch = tf.convert_to_tensor(y_labeled_batch)

                # train step
                self._train_step(x_labeled_batch, x_unlabeled_batch, y_labeled_batch)

            # put stats in dictionary (easy management)
            train_results = {
                self._loss1.name: self._loss1.result(),
                self._loss2.name: self._loss2.result(),
                self._acc1.name: self._acc1.result(),
                self._acc2.name: self._acc2.result(),
            }
            results = {'train': train_results}

            # test on validation set
            if x_val is not None and y_val is not None:
                val_results = self.test(x_val, y_val, evaluation_mapping=evaluation_mapping)
                results['val'] = val_results

            # save logs
            with train_summary_writer.as_default():
                for results_ in results.values():
                    for k, v in results_:
                        tf.summary.scalar(k, v, step=epoch)

            # save checkpoint
            if checkpoint is not None:
                checkpoint.save(file_prefix=checkpoints_path)

            # print stats
            for dataset, results_ in results.items():
                print(f'Epoch {epoch + 1} - ', dataset, ' - ', sep='', end='')
                for k, v in results_:
                    print(k, '=', v, ', ', sep='', end='')
                print()

            # reset losses and metrics
            self._loss1.reset_states()
            self._loss2.reset_states()
            self._loss1_cls.reset_states()
            self._loss2_cls.reset_states()
            self._loss1_con.reset_states()
            self._loss2_con.reset_states()
            self._loss1_sta.reset_states()
            self._loss2_sta.reset_states()
            self._acc1.reset_states()
            self._acc2.reset_states()

    @tf.function
    def _train_step(self, x_labeled, x_unlabeled, y_labeled):
        # noisy augmented batches (TODO: improvement with data augmentation instead of noise)
        B1_labeled = self._noisy_augment(x_labeled)
        B2_labeled = self._noisy_augment(x_labeled)
        B1_unlabeled = self._noisy_augment(x_unlabeled)
        B2_unlabeled = self._noisy_augment(x_unlabeled)

        # compute masks (to remove padding)
        mask_labeled = self.mask.compute_mask(x_labeled)
        mask_unlabeled = self.mask.compute_mask(x_unlabeled)

        # remove padding from labels
        y_labeled = y_labeled[mask_labeled]

        with tf.GradientTape(persistent=True) as tape:
            # predict augmented labeled samples (for classification constraint)
            prob1_labeled = self(B1_labeled, training=True, student='student1')
            prob2_labeled = self(B2_labeled, training=True, student='student2')

            # predict augmented unlabeled samples (for consistency and stabilization constraints)
            prob1_unlabeled_B1 = self.student1(B1_unlabeled, training=True)
            prob1_unlabeled_B2 = self.student1(B2_unlabeled, training=True)
            prob2_unlabeled_B1 = self.student2(B1_unlabeled, training=True)
            prob2_unlabeled_B2 = self.student2(B2_unlabeled, training=True)

            # remove padding
            prob1_labeled = prob1_labeled[mask_labeled]
            prob2_labeled = prob2_labeled[mask_labeled]
            prob1_unlabeled_B1 = prob1_unlabeled_B1[mask_unlabeled]
            prob1_unlabeled_B2 = prob1_unlabeled_B2[mask_unlabeled]
            prob2_unlabeled_B1 = prob2_unlabeled_B1[mask_unlabeled]
            prob2_unlabeled_B2 = prob2_unlabeled_B2[mask_unlabeled]

            # compute classification losses
            L1_cls = self._loss_cls(y_labeled, prob1_labeled)
            L2_cls = self._loss_cls(y_labeled, prob2_labeled)

            # compute consistency losses
            L1_con = self._loss_con(prob1_unlabeled_B1, prob1_unlabeled_B2)
            L2_con = self._loss_con(prob2_unlabeled_B1, prob2_unlabeled_B2)

            # prediction
            P1_unlabeled_B1 = tf.argmax(prob1_unlabeled_B1, axis=-1)
            P1_unlabeled_B2 = tf.argmax(prob1_unlabeled_B2, axis=-1)
            P2_unlabeled_B1 = tf.argmax(prob2_unlabeled_B1, axis=-1)
            P2_unlabeled_B2 = tf.argmax(prob2_unlabeled_B2, axis=-1)

            # confidence (probability of predicted class)
            M1_unlabeled_B1 = tf.reduce_max(prob1_unlabeled_B1, axis=-1)
            M1_unlabeled_B2 = tf.reduce_max(prob1_unlabeled_B2, axis=-1)
            M2_unlabeled_B1 = tf.reduce_max(prob2_unlabeled_B1, axis=-1)
            M2_unlabeled_B2 = tf.reduce_max(prob2_unlabeled_B2, axis=-1)

            # stable samples (masks to index probabilities)
            R1 = tf.logical_and(P1_unlabeled_B1 == P1_unlabeled_B2,
                                tf.logical_or(M1_unlabeled_B1 > self.epsilon, M1_unlabeled_B2 > self.epsilon))
            R2 = tf.logical_and(P2_unlabeled_B1 == P2_unlabeled_B2,
                                tf.logical_or(M2_unlabeled_B1 > self.epsilon, M2_unlabeled_B2 > self.epsilon))
            R12 = tf.logical_and(R1, R2)

            # stabilities
            epsilon1 = MSE(prob1_unlabeled_B1[R12], prob1_unlabeled_B2[R12])
            epsilon2 = MSE(prob2_unlabeled_B1[R12], prob2_unlabeled_B2[R12])

            # compute stabilization losses
            L1_sta = self._loss_sta(prob1_unlabeled_B1[R12][epsilon1 > epsilon2],
                                    prob2_unlabeled_B1[R12][epsilon1 > epsilon2])
            L2_sta = self._loss_sta(prob1_unlabeled_B2[R12][epsilon1 < epsilon2],
                                    prob2_unlabeled_B2[R12][epsilon1 < epsilon2])

            L1_sta += self._loss_sta(prob1_unlabeled_B1[tf.logical_and(tf.logical_not(R1), R2)],
                                     prob2_unlabeled_B1[tf.logical_and(tf.logical_not(R1), R2)])
            L2_sta += self._loss_sta(prob1_unlabeled_B2[tf.logical_and(R1, tf.logical_not(R2))],
                                     prob2_unlabeled_B2[tf.logical_and(R1, tf.logical_not(R2))])

            # compute complete losses
            L1 = L1_cls + self._lambda1 * L1_con + self._lambda2 * L1_sta
            L2 = L2_cls + self._lambda1 * L2_con + self._lambda2 * L2_sta

        gradients1 = tape.gradient(L1, self.student1.trainable_variables)
        gradients2 = tape.gradient(L2, self.student2.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients1, self.student1.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients2, self.student2.trainable_variables))
        del tape  # to release memory (persistent tape)

        # update loss
        self._loss1.update_state(L1)
        self._loss2.update_state(L2)
        self._loss1_cls.update_state(L1_cls)
        self._loss2_cls.update_state(L2_cls)
        self._loss1_con.update_state(L1_con)
        self._loss2_con.update_state(L2_con)
        self._loss1_sta.update_state(L1_sta)
        self._loss2_sta.update_state(L2_sta)

        # update accuracy
        self._acc1.update_state(y_labeled, prob1_labeled)
        self._acc2.update_state(y_labeled, prob2_labeled)

    def test(self, x, y, batch_size=32, evaluation_mapping=None):
        """
        Computes loss, accuracy and phone error rate for a dataset. The dataset could also be big and will be split in
        batches.

        :param x: numpy array of variable-length numpy arrays of shape (n_frames,n_features), features
        :param y: numpy array of variable-length numpy arrays of shape (n_frames,), labels
        :param batch_size: batch size
        :param evaluation_mapping: dictionary {training label -> test label}, the test phones should be a subset of the
            training phones
        :return: dictionary {metric_name -> value}
        """
        results = {}

        # predict
        y_pred1, mask = self.pad_and_predict(x, student='student1', batch_size=batch_size)
        y_pred2, _ = self.pad_and_predict(x, student='student2', batch_size=batch_size)

        # remove padding
        y_pred1 = y_pred1[mask]
        y_pred2 = y_pred2[mask]

        # loss
        results['loss1'] = self._loss_cls(y, y_pred1)
        results['loss2'] = self._loss_cls(y, y_pred2)

        # accuracy on original phones
        results['acc1'] = SparseCategoricalAccuracy()(y, y_pred1)
        results['acc2'] = SparseCategoricalAccuracy()(y, y_pred2)

        # accuracy on test phones
        y_pred1 = np.argmax(y_pred1, axis=-1)
        y_pred2 = np.argmax(y_pred2, axis=-1)
        if evaluation_mapping is not None:
            y_pred1 = map_labels(evaluation_mapping, y_pred1)
            y_pred2 = map_labels(evaluation_mapping, y_pred2)
            results['acc1_test_phones'] = Accuracy()(y, y_pred1)
            results['acc2_test_phones'] = Accuracy()(y, y_pred2)

        # phone error rate (for original phones if evaluation mapping is None, otherwise on test phones)
        results['per1'] = PhoneErrorRate()(y, y_pred1)
        results['per2'] = PhoneErrorRate()(y, y_pred2)

        return results
