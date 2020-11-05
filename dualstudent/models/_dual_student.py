import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Masking, Bidirectional, LSTM, Dense
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError, MSE
from tensorflow.keras.metrics import Mean


class DualStudent(Model):
    """"
    Dual Student for Automatic Speech Recognition (ASR).

    Original proposal for image classification: https://arxiv.org/abs/1909.01804
    """

    def __init__(self, n_classes, n_features, n_units=768, n_hidden_layers=5, epsilon=0.016395, lambda1=1, lambda2=100,
                 padding_value=0, student_version='mono_directional'):
        """
        Constructs a Dual Student model.

        :param n_classes: number of classes (i.e. number of units in the last layer)
        :param n_units: number of units for each hidden layer
        :param n_features: number of features per frame (i.e. last dimension)
        :param n_hidden_layers: number of layers
        :param epsilon: threshold for stable sample
        :param lambda1: weight of consistency constraint
        :param lambda2: weight of stabilization constraint
        :param padding_value: value used to pad input sequences (used as mask_value for Masking layer)
        :param student_version: one of:
            - 'mono_directional': both students have mono-directional LSTM layers
            - 'bidirectional: both students have bidirectional LSTM layers
            - 'imbalanced': one student has mono-directional LSTM layers, the other one bidirectional
        """
        super(DualStudent, self).__init__()

        self.n_classes = n_classes
        self.padding_value = padding_value
        self._n_features = n_features
        self._n_units = n_units
        self._n_hidden_layers = n_hidden_layers
        self._epsilon = epsilon
        self._lambda1 = lambda1
        self._lambda2 = lambda2

        # training losses
        self._cce = CategoricalCrossentropy()
        self._mse = MeanSquaredError()
        self._loss1 = Mean(name='loss1')  # TODO: why mean? and it is in tf.keras.metrics not tf.keras.losses
        self._loss2 = Mean(name='loss2')

        # compose students
        if student_version == 'mono_directional':
            lstm_types = ['mono_directional', 'mono_directional']
        elif student_version == 'bidirectional':
            lstm_types = ['bidirectional', 'bidirectional']
        elif student_version == 'imbalanced':
            lstm_types = ['mono_directional', 'bidirectional']
        else:
            raise ValueError('Invalid student version')
        self.student1 = self._get_student('student1', lstm_types[0])
        self.student2 = self._get_student('student2', lstm_types[1])

    def _get_student(self, name, lstm_type):
        student = Sequential(name=name)
        student.add(Masking(mask_value=self.padding_value))
        if lstm_type == 'mono_directional':
            for i in range(self._n_hidden_layers):
                student.add(LSTM(units=self._n_units, return_sequences=True))
        elif lstm_type == 'bidirectional':
            for i in range(self._n_hidden_layers):
                student.add(Bidirectional(LSTM(units=self._n_units, return_sequences=True)))
        else:
            raise ValueError('Invalid LSTM version')
        student.add(Dense(units=self.n_classes, activation="softmax"))
        return student

    def call(self, inputs, student='both', **kwargs):
        """
        TODO: do we really want to do in this way? I think a caller wants just to get the answer... we could return the most confident one
        the internal calls to the individual students can be done with self.students['student_x'](inputs)

        :param inputs:
        :param student:
        :param kwargs:
        :return:
        """
        if student == 'both':
            return self.call(inputs, 'student1'), self.call(inputs, 'student2')
        else:
            return self.students[student](inputs)

    def train(self, x_labeled, x_unlabeled, y, x_val=None, y_val=None, n_epochs=1, batch_size=32, shuffle=True):
        # TODO: cross-batch statefulness? one update for 758 samples will be slow... maybe we have to split the sequences in sub-sequences of 20 samples
        labeled_batch_size = int(len(x_labeled) / (len(x_unlabeled) + len(x_labeled)) * batch_size)
        unlabeled_batch_size = batch_size - labeled_batch_size
        n_batches = min(int(len(x_unlabeled) / unlabeled_batch_size), int(len(x_labeled) / labeled_batch_size))

        for epoch in range(n_epochs):
            if shuffle:
                indices = np.arange(len(x_labeled))  # get indices to shuffle coherently features and labels
                np.random.shuffle(indices)
                x_labeled = x_labeled[indices]
                y = y[indices]
                np.random.shuffle(x_unlabeled)

            for i in range(n_batches):
                x_labeled_batch = x_labeled[i * labeled_batch_size:(i + 1) * labeled_batch_size]
                y_batch = y[i * labeled_batch_size:(i + 1) * labeled_batch_size]
                x_unlabeled_batch = x_unlabeled[i * unlabeled_batch_size:(i + 1) * unlabeled_batch_size]

                L1, L2 = self._train_step(x_labeled_batch, x_unlabeled_batch, y_batch)
            # this is not the loss for the whole epoch it is just to test if it works
            print("epoch loss", L1, L2)

            # self.test_step(x_val, y_val)  # TODO

    @tf.function
    def _train_step(self, x_labeled, x_unlabeled, y):
        # noisy augmented batches (TODO: improvement with data augmentation instead of noise)
        B1_labeled = x_labeled + tf.random.normal(shape=x_labeled.shape)
        B2_labeled = x_labeled + tf.random.normal(shape=x_labeled.shape)
        B1_unlabeled = x_unlabeled + tf.random.normal(shape=x_unlabeled.shape)
        B2_unlabeled = x_unlabeled + tf.random.normal(shape=x_unlabeled.shape)

        with tf.GradientTape(persistent=True) as tape:
            # predict augmented labeled samples (for classification constraint)
            prob1_labeled = self.student1(B1_labeled, training=True)
            prob2_labeled = self.student2(B2_labeled, training=True)

            # predict augmented unlabeled samples (for consistency and stabilization constraints)
            prob1_unlabeled_B1 = self.student1(B1_unlabeled, training=True)
            prob1_unlabeled_B2 = self.student1(B2_unlabeled, training=True)
            prob2_unlabeled_B1 = self.student2(B1_unlabeled, training=True)
            prob2_unlabeled_B2 = self.student2(B2_unlabeled, training=True)

            # compute classification losses
            L1_cls = self._cce(y, prob1_labeled)
            L2_cls = self._cce(y, prob2_labeled)

            # compute consistency losses
            L1_con = self._mse(prob1_unlabeled_B1, prob1_unlabeled_B2)
            L2_con = self._mse(prob2_unlabeled_B1, prob2_unlabeled_B2)

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
                                tf.logical_or(M1_unlabeled_B1 > self._epsilon, M1_unlabeled_B2 > self._epsilon))
            R2 = tf.logical_and(P2_unlabeled_B1 == P2_unlabeled_B2,
                                tf.logical_or(M2_unlabeled_B1 > self._epsilon, M2_unlabeled_B2 > self._epsilon))
            R12 = tf.logical_and(tf.equal(R1, True), tf.equal(R2, True))

            # stabilities
            epsilon1 = MSE(prob1_unlabeled_B1[R12], prob1_unlabeled_B2[R12])
            epsilon2 = MSE(prob2_unlabeled_B1[R12], prob2_unlabeled_B2[R12])

            # compute stabilization losses
            L1_sta = self._mse(prob1_unlabeled_B1[R12][epsilon1 > epsilon2],
                               prob2_unlabeled_B1[R12][epsilon1 > epsilon2])
            L2_sta = self._mse(prob1_unlabeled_B2[R12][epsilon1 < epsilon2],
                               prob2_unlabeled_B2[R12][epsilon1 < epsilon2])

            L1_sta += self._mse(prob1_unlabeled_B1[tf.logical_and(tf.equal(R1, False), tf.equal(R2, True))],
                                prob2_unlabeled_B1[tf.logical_and(tf.equal(R1, False), tf.equal(R2, True))])
            L2_sta += self._mse(prob1_unlabeled_B2[tf.logical_and(tf.equal(R1, True), tf.equal(R2, False))],
                                prob2_unlabeled_B2[tf.logical_and(tf.equal(R1, True), tf.equal(R2, False))])

            # compute complete losses
            L1 = L1_cls + self._lambda1 * L1_con + self._lambda2 * L1_sta
            L2 = L2_cls + self._lambda1 * L2_con + self._lambda2 * L2_sta

        gradients1 = tape.gradient(L1, self.student1.trainable_variables)
        gradients2 = tape.gradient(L2, self.student2.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients1, self.student1.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients2, self.student2.trainable_variables))
        del tape  # to release memory (persistent tape)

        # TODO: update metrics

        return L1, L2

    @tf.function
    def _test_step(self, data):
        # TODO
        pass
