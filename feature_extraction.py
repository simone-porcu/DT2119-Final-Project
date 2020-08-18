"""
This module contains utilities for extracting features from the TIMIT dataset.

Authors: Franco Ruggeri, Andrea Caraffa
"""

import os
import numpy as np
from pysndfile import sndio


root_dir = 'data/timit'


def path_to_info(path):
    """
    Extracts the information about an utterance of the TIMIT dataset starting from its path.

    Path format: <USAGE>/<DIALECT>/<SEX><SPEAKER_ID>/<SENTENCE_ID>.<FILE_TYPE>
    Example: timit/train/dr1/mwar0/sx415.wav

    See timit/readme.doc for an explanation of each field.
    """
    path, aux = os.path.split(path)
    sentence_id, file_type = aux.split(sep='.')
    path, aux = os.path.split(path)
    sex = aux[:1]
    speaker_id = aux[1:]
    path, dialect = os.path.split(path)
    path, usage = os.path.split(path)
    return {'usage': usage, 'dialect': dialect, 'sex': sex, 'speaker_id': speaker_id, 'sentence_id': sentence_id,
            'file_type': file_type}


def load_audio(path):
    """
    Loads audio data from file using pysndfile

    Note that, by default pysndfile converts the samples into floating point
    numbers and rescales them in the range [-1, 1]. This is avoided by specifying
    the option dtype=np.int16 which keeps both the original data type and range
    of values.

    Source: lab3 by Giampiero Salvi (slightly modified)
    """
    path = os.path.join(root_dir, path)
    snd_obj = sndio.read(path, dtype=np.int16)
    samples = np.array(snd_obj[0])
    sampling_rate = snd_obj[1]
    return {'samples': samples, 'sampling_rate': sampling_rate}


if __name__ == '__main__':
    print(path_to_info('train/dr1/mwar0/sx415.wav'))
    print(load_audio('train/dr1/mwar0/sx415.wav'))



# # TODO add mapping features/target labels
# def featuresExtraction(filename):
#   rest, mode = os.path.split(filename)
#   if mode == 'train':
#     tot = 3696
#   else:
#     tot = 192
#
#   count = 1
#   data = []
#   for root, dirs, files in os.walk(speechDirectory+filename):
#     for file in files:
#       if file.endswith('.wav') and not file.startswith('SA'):
#         filename = os.path.join(root, file)
#
#         if count % 500 == 0:
#           print(float(count)/tot, "%")
#           count = count + 1
#
#         samples, samplingrate = load_audio(filename)
#
#         mfcc_feat = mfcc(samples, samplingrate)
#         delta_feat = delta(mfcc_feat,3)
#         delta_delta_feat = delta(delta_feat,3)
#
#         feat = np.hstack((mfcc_feat, delta_feat, delta_delta_feat))
#
#         """
#         from lab 3
#         wordTrans = list(path2info(filename)[2])
#         phoneTrans = words2phones(wordTrans, prondict)
#         targets = forcedAlignment(lmfcc,phoneHMMs, phoneTrans)
#         targets = [stateList.index(target) for target in targets]
#         """
#
#         data.append({'filename': filename, 'features': feat, 'targets': targets})
#
#   print(float(count)/tot, "%")
#
#   return data
#
#
# # phone_map_tsv is the 60-48-39.map file
# def phoneMapping(phone_map_tsv):
#   df = pd.read_csv(phone_map_tsv, sep="\t", index_col=0)
#   df = df.dropna()
#   df = df.drop('eval', axis=1)
#   train_phn_idx = {k: i for i, k in enumerate(df['train'].unique())}
#   df['train_idx'] = df['train'].map(train_phn_idx)
#   phone_to_idx = df['train_idx'].to_dict()
#
#   # train_phn_idx: mapping from 48 phonemes to 48 idx
#   # phone_to_idx:  mapping from 61 phonemes to 48 idx
#   return train_phn_idx, phone_to_idx