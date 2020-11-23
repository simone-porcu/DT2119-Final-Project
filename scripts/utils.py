import numpy as np
from dualstudent.datasets import timit

N_HIDDEN_LAYERS = 1
N_UNITS = 2
PADDING_VALUE = np.inf


class Config:
    def __init__(self, version, n_hidden_layers, n_units, n_epochs, batch_size, unlabeled_percentage, optimizer,
                 consistency_loss, consistency_scale, stabilization_scale, xi, sigma, schedule, schedule_length,
                 normalization, seed):
        self.version = version
        self.n_hidden_layers = n_hidden_layers
        self.n_units = n_units
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.unlabeled_percentage = unlabeled_percentage
        self.optimizer = optimizer
        self.consistency_loss = consistency_loss
        self.consistency_scale = consistency_scale
        self.stabilization_scale = stabilization_scale
        self.xi = xi
        self.sigma = sigma
        self.schedule = schedule
        self.schedule_length = schedule_length
        self.normalization = normalization
        self.seed = seed

    def load_from_name(self, model_name):
        aux = model_name.split('-')
        self.version = aux[0]
        self.n_hidden_layers = int(aux[1])
        self.n_units = int(aux[2])
        self.n_epochs = int(aux[3])
        self.batch_size = int(aux[4])
        self.unlabeled_percentage = float(aux[5])
        self.optimizer = aux[6]
        self.consistency_loss = aux[7]
        self.consistency_scale = float(aux[8])
        self.stabilization_scale = float(aux[9])
        self.xi = float(aux[10])
        self.sigma = float(aux[11])
        self.schedule = aux[12]
        self.schedule_length = int(aux[13])
        self.normalization = aux[14]
        self.seed = aux[15]

    def __str__(self):
        return (
            str(self.version) + '-' +
            str(self.n_hidden_layers) + '-' +
            str(self.n_units) + '-' +
            str(self.n_epochs) + '-' +
            str(self.batch_size) + '-' +
            str(self.unlabeled_percentage) + '-' +
            str(self.optimizer) + '-' +
            str(self.consistency_loss) + '-' +
            str(self.consistency_scale) + '-' +
            str(self.stabilization_scale) + '-' +
            str(self.xi) + '-' +
            str(self.sigma) + '-' +
            str(self.schedule) + '-' +
            str(self.schedule_length) + '-' +
            str(self.normalization) + '-' +
            str(self.seed)
        )


def get_number_of_classes():
    return len(timit.get_phone_mapping()[1])
