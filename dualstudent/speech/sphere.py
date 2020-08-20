import numpy as np
from pysndfile import sndio


def load_audio(path):
    """
    Loads the utterance samples from a file.

    Source: lab3 of DT2119 Speech and Speaker Recognition at KTH, by prof. Giampiero Salvi (slightly modified)

    :param path: path to the utterance file (.wav)
    :return: (samples, sample rate)
    """
    snd_obj = sndio.read(path, dtype=np.int16)
    samples = np.array(snd_obj[0])
    sample_rate = snd_obj[1]
    return samples, sample_rate
