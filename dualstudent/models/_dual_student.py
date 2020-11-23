import numpy as np
import tensorflow as tf
from pathlib import Path
from math import ceil
from tqdm import trange
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Masking, Bidirectional, LSTM, Dense
from tensorflow.keras.losses import KLDivergence, MSE
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean, Accuracy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dualstudent.metrics import PhoneErrorRate
from dualstudent.models._utils import sigmoid_rampup, triangular_cycling, sinusoidal_cycling, select_batch, map_labels


class DualStudent(Model):
    """"
    Dual Student for Automatic Speech Recognition (ASR).

    How to train: 1) set the optimizer by means of compile(), 2) use train()
    How to test: use test()

    Remarks:
    - Do not use fit() by Keras, use train()
    - Do not use evaluate() by Keras, use test()
    - Compiled metrics and loss (i.e. set by means of compile()) are not used

    Original proposal for image classification: https://arxiv.org/abs/1909.01804
    """

    def __init__(self, n_classes, n_hidden_layers=3, n_units=96, consistency_loss='mse', consistency_scale=10,
                 stabilization_scale=100, xi=0.6, padding_value=0., sigma=0.01, schedule='rampup',
                 schedule_length=5, version='mono_directional'):
        """
        Constructs a Dual Student model.

        :param n_classes: number of classes (i.e. number of units in the last layer of each student)
        :param n_hidden_layers: number of hidden layers in each student (i.e. LSTM layers)
        :param n_units: number of units for each hidden layer
        :param consistency_loss: one of 'mse', 'kl'
        :param consistency_scale: maximum value of weight for consistency constraint
        :param stabilization_scale: maximum value of weight for stabilization constraint
        :param xi: threshold for stable sample
        :param padding_value: value used to pad input sequences (used as mask_value for Masking layer)
        :param sigma: standard deviation for noisy augmentation
        :param schedule: type of schedule for lambdas, one of 'rampup', 'triangular_cycling', 'sinusoidal_cycling'
        :param schedule_length:
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
        self.xi = xi
        self.consistency_scale = consistency_scale
        self.stabilization_scale = stabilization_scale
        self.sigma = sigma
        self.version = version
        self.schedule = schedule
        self.schedule_length = schedule_length
        self._lambda1 = None
        self._lambda2 = None

        # schedule for lambdas
        if schedule == 'rampup':
            self.schedule_fn = sigmoid_rampup
        elif schedule == 'triangular_cycling':
            self.schedule_fn = triangular_cycling
        elif schedule == 'sinusoidal_cycling':
            self.schedule_fn = sinusoidal_cycling
        else:
            raise ValueError('Invalid schedule')

        # loss
        self._loss_cls = SparseCategoricalCrossentropy()            # classification loss
        self._loss_sta = MeanSquaredError()                         # stabilization loss
        if consistency_loss == 'mse':
            self._loss_con = MeanSquaredError()                     # consistency loss
        elif consistency_loss == 'kl':
            self._loss_con = KLDivergence()
        else:
            raise ValueError('Invalid consistency metric')

        # metrics for training
        self._loss1 = Mean(name='loss1')                            # we want to average the loss for each batch
        self._loss2 = Mean(name='loss2')
        self._loss1_cls = Mean(name='loss1_cls')
        self._loss2_cls = Mean(name='loss2_cls')
        self._loss1_con = Mean(name='loss1_con')
        self._loss2_con = Mean(name='loss2_con')
        self._loss1_sta = Mean(name='loss1_sta')
        self._loss2_sta = Mean(name='loss2_sta')
        self._acc1 = SparseCategoricalAccuracy(name='acc1')
        self._acc2 = SparseCategoricalAccuracy(name='acc2')

        # metrics for testing
        self._test_loss1 = Mean(name='test_loss1')
        self._test_loss2 = Mean(name='test_loss2')
        self._test_acc1_train_phones = SparseCategoricalAccuracy(name='test_acc1_train_phones')
        self._test_acc2_train_phones = SparseCategoricalAccuracy(name='test_acc2_train_phones')
        self._test_acc1 = Accuracy(name='test_acc1')
        self._test_acc2 = Accuracy(name='test_acc2')
        self._test_per1 = PhoneErrorRate(name='test_per1')
        self._test_per2 = PhoneErrorRate(name='test_per2')

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

        # masking layer (just to use compute_mask and remove padding)
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

    def build(self, input_shape):
        super(DualStudent, self).build(input_shape)
        self.student1.build(input_shape)
        self.student2.build(input_shape)

    def train(self, x_labeled, x_unlabeled, y_labeled, x_val=None, y_val=None, n_epochs=10, batch_size=32, shuffle=True,
              evaluation_mapping=None, logs_path=None, checkpoints_path=None, initial_epoch=0, seed=None):
        """
        Trains the students with both labeled and unlabeled data (semi-supervised learning).

        :param x_labeled: numpy array of numpy arrays (n_frames, n_features), features corresponding to y_labeled.
            'n_frames' can vary, padding is added to make x_labeled a tensor.
        :param x_unlabeled: numpy array of numpy arrays of shape (n_frames, n_features), features without labels.
            'n_frames' can vary, padding is added to make x_unlabeled a tensor.
        :param y_labeled: numpy array of numpy arrays of shape (n_frames,), labels corresponding to x_labeled.
            'n_frames' can vary, padding is added to make y_labeled a tensor.
        :param x_val: like x_labeled, but for validation set
        :param y_val: like y_labeled, but for validation set
        :param n_epochs: integer, number of training epochs
        :param batch_size: integer, batch size
        :param shuffle: boolean, whether to shuffle at each epoch or not
        :param evaluation_mapping: dictionary {training label -> test label}, the test phones should be a subset of the
            training phones
        :param logs_path: path where to save logs for TensorBoard
        :param checkpoints_path: path to a directory. If the directory contains checkpoints, the latest checkpoint is
            restored.
        :param initial_epoch: int, initial epoch from which to start the training. It can be used together with
            checkpoints_path to resume the training from a previous run.
        :param seed: seed for the random number generator
        """
        # set seed
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # show summary
        self.build(input_shape=(None,) + x_labeled[0].shape)
        self.student1.summary()
        self.student2.summary()

        # setup for logs
        train_summary_writer = None
        if logs_path is not None:
            train_summary_writer = tf.summary.create_file_writer(logs_path)

        # setup for checkpoints
        checkpoint = None
        if checkpoints_path is not None:
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
            checkpoint_path = tf.train.latest_checkpoint(checkpoints_path)
            if checkpoint_path is not None:
                checkpoint.restore(checkpoint_path)
            checkpoint_path = Path(checkpoints_path) / 'ckpt'
            checkpoint_path = str(checkpoint_path)

        # compute batch sizes
        labeled_batch_size = ceil(len(x_labeled) / (len(x_unlabeled) + len(x_labeled)) * batch_size)
        unlabeled_batch_size = batch_size - labeled_batch_size
        n_batches = min(ceil(len(x_unlabeled) / unlabeled_batch_size), ceil(len(x_labeled) / labeled_batch_size))

        # training loop
        for epoch in trange(initial_epoch, n_epochs, desc='epochs'):
            # ramp up lambda1 and lambda2
            self._lambda1 = self.consistency_scale * self.schedule_fn(epoch, self.schedule_length)
            self._lambda2 = self.stabilization_scale * self.schedule_fn(epoch, self.schedule_length)

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

            # put metrics in dictionary (easy management)
            train_metrics = {
                self._loss1.name: self._loss1.result(),
                self._loss2.name: self._loss2.result(),
                self._loss1_cls.name: self._loss1_cls.result(),
                self._loss2_cls.name: self._loss2_cls.result(),
                self._loss1_con.name: self._loss1_con.result(),
                self._loss2_con.name: self._loss2_con.result(),
                self._loss1_sta.name: self._loss1_sta.result(),
                self._loss2_sta.name: self._loss2_sta.result(),
                self._acc1.name: self._acc1.result(),
                self._acc2.name: self._acc2.result(),
            }
            metrics = {'train': train_metrics}

            # test on validation set
            if x_val is not None and y_val is not None:
                val_metrics = self.test(x_val, y_val, evaluation_mapping=evaluation_mapping)
                metrics['val'] = val_metrics

            # print metrics
            for dataset, metrics_ in metrics.items():
                print(f'Epoch {epoch + 1} - ', dataset, ' - ', sep='', end='')
                for k, v in metrics_.items():
                    print(f'{k}: {v}, ', end='')
                print()

            # save logs
            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    for dataset, metrics_ in metrics.items():
                        for k, v in metrics_.items():
                            tf.summary.scalar(k, v, step=epoch)

            # save checkpoint
            if checkpoint is not None:
                checkpoint.save(file_prefix=checkpoint_path)

            # reset metrics
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

    """
    If you want to use graph execution, pad the whole dataset externally and uncomment the decorator below.
    If you uncomment the decorator without padding the dataset, the graph will be compiled for each batch, 
    because train() pads at batch level and so the batches have different shapes. This would result in worse
    performance compared to eager execution.
    """
    # @tf.function
    def _train_step(self, x_labeled, x_unlabeled, y_labeled):
        # noisy augmented batches (TODO: improvement with data augmentation instead of noise)
        B1_labeled = self._noisy_augment(x_labeled)
        B2_labeled = self._noisy_augment(x_labeled)
        B1_unlabeled = self._noisy_augment(x_unlabeled)
        B2_unlabeled = self._noisy_augment(x_unlabeled)

        # compute masks (to remove padding)
        mask_labeled = self.mask.compute_mask(x_labeled)
        mask_unlabeled = self.mask.compute_mask(x_unlabeled)
        y_labeled = y_labeled[mask_labeled]     # remove padding from labels

        # forward pass
        with tf.GradientTape(persistent=True) as tape:
            # predict augmented labeled samples (for classification and consistency constraint)
            prob1_labeled_B1 = self.student1(B1_labeled, training=True)
            prob1_labeled_B2 = self.student1(B2_labeled, training=True)
            prob2_labeled_B1 = self.student2(B1_labeled, training=True)
            prob2_labeled_B2 = self.student2(B2_labeled, training=True)

            # predict augmented unlabeled samples (for consistency and stabilization constraints)
            prob1_unlabeled_B1 = self.student1(B1_unlabeled, training=True)
            prob1_unlabeled_B2 = self.student1(B2_unlabeled, training=True)
            prob2_unlabeled_B1 = self.student2(B1_unlabeled, training=True)
            prob2_unlabeled_B2 = self.student2(B2_unlabeled, training=True)

            # remove padding
            prob1_labeled_B1 = prob1_labeled_B1[mask_labeled]
            prob1_labeled_B2 = prob1_labeled_B2[mask_labeled]
            prob2_labeled_B1 = prob2_labeled_B1[mask_labeled]
            prob2_labeled_B2 = prob2_labeled_B2[mask_labeled]
            prob1_unlabeled_B1 = prob1_unlabeled_B1[mask_unlabeled]
            prob1_unlabeled_B2 = prob1_unlabeled_B2[mask_unlabeled]
            prob2_unlabeled_B1 = prob2_unlabeled_B1[mask_unlabeled]
            prob2_unlabeled_B2 = prob2_unlabeled_B2[mask_unlabeled]

            # compute classification losses
            L1_cls = self._loss_cls(y_labeled, prob1_labeled_B1)
            L2_cls = self._loss_cls(y_labeled, prob2_labeled_B2)

            # concatenate labeled and unlabeled probability predictions (for consistency loss)
            prob1_labeled_unlabeled_B1 = tf.concat([prob1_labeled_B1, prob1_unlabeled_B1], axis=0)
            prob1_labeled_unlabeled_B2 = tf.concat([prob1_labeled_B2, prob1_unlabeled_B2], axis=0)
            prob2_labeled_unlabeled_B1 = tf.concat([prob2_labeled_B1, prob2_unlabeled_B1], axis=0)
            prob2_labeled_unlabeled_B2 = tf.concat([prob2_labeled_B2, prob2_unlabeled_B2], axis=0)

            # compute consistency losses
            L1_con = self._loss_con(prob1_labeled_unlabeled_B1, prob1_labeled_unlabeled_B2)
            L2_con = self._loss_con(prob2_labeled_unlabeled_B1, prob2_labeled_unlabeled_B2)

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
                                tf.logical_or(M1_unlabeled_B1 > self.xi, M1_unlabeled_B2 > self.xi))
            R2 = tf.logical_and(P2_unlabeled_B1 == P2_unlabeled_B2,
                                tf.logical_or(M2_unlabeled_B1 > self.xi, M2_unlabeled_B2 > self.xi))
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

        # backward pass
        gradients1 = tape.gradient(L1, self.student1.trainable_variables)
        gradients2 = tape.gradient(L2, self.student2.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients1, self.student1.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients2, self.student2.trainable_variables))
        del tape  # to release memory (persistent tape)

        # update metrics
        self._loss1.update_state(L1)
        self._loss2.update_state(L2)
        self._loss1_cls.update_state(L1_cls)
        self._loss2_cls.update_state(L2_cls)
        self._loss1_con.update_state(L1_con)
        self._loss2_con.update_state(L2_con)
        self._loss1_sta.update_state(L1_sta)
        self._loss2_sta.update_state(L2_sta)
        self._acc1.update_state(y_labeled, prob1_labeled_B1)
        self._acc2.update_state(y_labeled, prob2_labeled_B2)

    def test(self, x, y, batch_size=32, evaluation_mapping=None):
        """
        Tests the model (both students).

        :param x: numpy array of numpy arrays (n_frames, n_features), features corresponding to y_labeled.
            'n_frames' can vary, padding is added to make x a tensor.
        :param y: numpy array of numpy arrays of shape (n_frames,), labels corresponding to x_labeled.
            'n_frames' can vary, padding is added to make y a tensor.
        :param batch_size: integer, batch size
        :param evaluation_mapping: dictionary {training label -> test label}, the test phones should be a subset of the
            training phones
        :return: dictionary {metric_name -> value}
        """
        # test batch by batch
        n_batches = ceil(len(x) / batch_size)
        for i in trange(n_batches, desc='test batches'):
            # select batch
            x_batch = select_batch(x, i, batch_size)
            y_batch = select_batch(y, i, batch_size)

            # pad batch
            x_batch = pad_sequences(x_batch, padding='post', value=self.padding_value, dtype='float32')
            y_batch = pad_sequences(y_batch, padding='post', value=-1)

            # convert to tensors
            x_batch = tf.convert_to_tensor(x_batch)
            y_batch = tf.convert_to_tensor(y_batch)

            # test step
            self._test_step(x_batch, y_batch, evaluation_mapping)

        # put metrics in dictionary (easy management)
        test_metrics = {
            self._test_loss1.name: self._test_loss1.result(),
            self._test_loss2.name: self._test_loss2.result(),
            self._test_acc1_train_phones.name: self._test_acc1_train_phones.result(),
            self._test_acc2_train_phones.name: self._test_acc2_train_phones.result(),
            self._test_acc1.name: self._test_acc1.result(),
            self._test_acc2.name: self._test_acc2.result(),
            self._test_per1.name: self._test_per1.result(),
            self._test_per2.name: self._test_per2.result(),
        }

        # reset metrics
        self._test_loss1.reset_states()
        self._test_loss2.reset_states()
        self._test_acc1_train_phones.reset_states()
        self._test_acc2_train_phones.reset_states()
        self._test_acc1.reset_states()
        self._test_acc2.reset_states()
        self._test_per1.reset_states()
        self._test_per2.reset_states()

        return test_metrics

    # @tf.function      # see note in _train_step()
    def _test_step(self, x, y, evaluation_mapping):
        # compute mask (to remove padding)
        mask = self.mask.compute_mask(x)

        # forward pass
        y_prob1_train_phones = self.student1(x, training=False)
        y_prob2_train_phones = self.student2(x, training=False)
        y_pred1_train_phones = tf.argmax(y_prob1_train_phones, axis=-1)
        y_pred2_train_phones = tf.argmax(y_prob2_train_phones, axis=-1)
        y_train_phones = tf.identity(y)

        # map labels to set of test phones
        if evaluation_mapping is not None:
            y = tf.numpy_function(map_labels, [y_train_phones, evaluation_mapping], [tf.float32])
            y_pred1 = tf.numpy_function(map_labels, [y_pred1_train_phones, evaluation_mapping], [tf.float32])
            y_pred2 = tf.numpy_function(map_labels, [y_pred2_train_phones, evaluation_mapping], [tf.float32])
        else:
            y = y_train_phones
            y_pred1 = y_pred1_train_phones
            y_pred2 = y_pred2_train_phones

        # update phone error rate
        self._test_per1.update_state(y, y_pred1, mask)
        self._test_per2.update_state(y, y_pred2, mask)

        # remove padding
        y_pred1 = y_pred1[mask]
        y_pred2 = y_pred2[mask]
        y_prob1_train_phones = y_prob1_train_phones[mask]
        y_prob2_train_phones = y_prob2_train_phones[mask]
        y_train_phones = y_train_phones[mask]
        y = y[mask]

        # compute loss
        loss1 = self._loss_cls(y_train_phones, y_prob1_train_phones)
        loss2 = self._loss_cls(y_train_phones, y_prob2_train_phones)

        # update loss
        self._test_loss1.update_state(loss1)
        self._test_loss2.update_state(loss2)

        # update accuracy using training phones
        self._test_acc1_train_phones.update_state(y_train_phones, y_prob1_train_phones)
        self._test_acc2_train_phones.update_state(y_train_phones, y_prob2_train_phones)

        # update accuracy using test phones
        self._test_acc1.update_state(y, y_pred1)
        self._test_acc2.update_state(y, y_pred2)
