from types import SimpleNamespace

import numpy as np


def binary_data(samples, vocabulary):
    vocabulary = {
        word: i
        for i, word
        in enumerate(vocabulary)
    }

    data = SimpleNamespace()
    data.x = np.zeros((len(samples), len(vocabulary)))
    for i, sample in enumerate(samples):
        data.x[i][[vocabulary[word] for word in sample.text]] = 1
    data.y = np.array([sample.label for sample in samples])

    return data


def fasttext_data(samples, vocabulary):

    vocabulary = set(vocabulary)
    embeddings = dict()

    with open('data/sk.vec') as f:
        for line in f:
            word, emb = line.strip().split(maxsplit=1)
            if word in vocabulary:
                embeddings[word] = np.array([float(scalar) for scalar in emb.split()])

    data = SimpleNamespace()
    data.x = np.array([
        np.mean(
            np.array([
                embeddings[token]
                for token
                in sample.text
                if token in embeddings
            ]),
            axis=0
        )
        for sample
        in samples
    ])
    data.y = np.array([sample.label for sample in samples])

    return data


def elmo_data(samples):

    from elmoformanylangs import Embedder
    elmo = Embedder('data/elmo_model', batch_size=1)

    data = SimpleNamespace()
    data.x = elmo.sents2elmo([sample.text for sample in samples])
    data.x = np.array([
        np.mean(datum, axis=0)
        for datum
        in data.x
    ])
    data.y = np.array([sample.label for sample in samples])

    return data
