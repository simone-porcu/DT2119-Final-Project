import numpy as np
from dualstudent.datasets import timit

# tune here!
N_EPOCHS = 50
BATCH_SIZE = 32
UNLABELED_PERCENTAGE = 0.7
OPTIMIZER = 'adam'              # one of 'adam', 'adam_w', 'sgd', 'sgd_w'
VERSION = 'mono_directional'    # one of 'mono_directional', 'bidirectional', 'imbalanced'
N_HIDDEN_LAYERS = 2
N_UNITS = 3
CONSISTENCY_SCALE = 0           # weight of consistency constraint
STABILIZATION_SCALE = 0         # weight of stabilization constraint
EPSILON = 0.6                   # confidence threshold
SIGMA = 0.01                    # standard deviation for noisy augmentation
SEED = 1
PADDING_VALUE = np.inf


class Config:
    def __init__(self, model_name=None):
        if model_name is None:
            self.version = VERSION
            self.n_hidden_layers = N_HIDDEN_LAYERS
            self.n_units = N_UNITS
            self.n_epochs = N_EPOCHS
            self.batch_size = BATCH_SIZE
            self.unlabeled_percentage = UNLABELED_PERCENTAGE
            self.optimizer = OPTIMIZER
            self.consistency_scale = CONSISTENCY_SCALE
            self.stabilization_scale = STABILIZATION_SCALE
            self.epsilon = EPSILON
            self.sigma = SIGMA
            self.seed = SEED
            self.padding_value = PADDING_VALUE
        else:
            aux = model_name.split('-')
            self.version = aux[0]
            self.n_hidden_layers = int(aux[1])
            self.n_units = int(aux[2])
            self.n_epochs = int(aux[3])
            self.batch_size = int(aux[4])
            self.unlabeled_percentage = float(aux[5])
            self.optimizer = aux[6]
            self.consistency_scale = float(aux[7])
            self.stabilization_scale = float(aux[8])
            self.epsilon = float(aux[9])
            self.sigma = float(aux[10])

    def __str__(self):
        return (
            str(self.version) + '-' +
            str(self.n_hidden_layers) + '-' +
            str(self.n_units) + '-' +
            str(self.n_epochs) + '-' +
            str(self.batch_size) + '-' +
            str(self.unlabeled_percentage) + '-' +
            str(self.optimizer) + '-' +
            str(self.consistency_scale) + '-' +
            str(self.stabilization_scale) + '-' +
            str(self.epsilon) + '-' +
            str(self.sigma)
        )


def get_number_of_classes():
    return len(timit.get_phone_mapping()[1])
