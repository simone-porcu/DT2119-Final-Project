from pathlib import Path


def load_transcription(filepath):
    """
    Loads the phonetic transcription of an utterance from a file.

    :param filepath: path to the transcription file (.phn)
    :return: list of tuples (begin_sample, end_sample, phone)
    """
    filepath = Path(filepath)
    with filepath.open() as f:
        lines = f.read().split('\n')
    transcription = map(lambda line: line.split(' '), lines)
    transcription = filter(lambda segment: len(segment) == 3, transcription)    # remove invalid lines
    transcription = map(lambda segment: (int(segment[0]), int(segment[1]), segment[2]), transcription)
    transcription = list(transcription)
    return transcription
