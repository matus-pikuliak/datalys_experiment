import random
from types import SimpleNamespace

import stanfordnlp
import unidecode


def load_samples(filename):

    samples = []
    with open(filename) as f:
        for line in f:
            sample = SimpleNamespace()
            sample.label, sample.text = line.split(maxsplit=1)
            sample.label = int(float(sample.label) > 0)
            samples.append(sample)

    random.seed(1)
    random.shuffle(samples)

    return samples


def preprocess_text(samples, lowercase=False, lemmatize=False, remove_diacritics=False):

    if lemmatize:
        pipeline = stanfordnlp.Pipeline(processors='tokenize,pos,lemma', lang='sk')
    else:
        pipeline = stanfordnlp.Pipeline(processors='tokenize', lang='sk')

    for i, sample in enumerate(samples):
        doc = pipeline(sample.text)

        if lemmatize:
            tokens = [word.lemma for sentence in doc.sentences for word in sentence.words]
        else:
            tokens = [word.text for sentence in doc.sentences for word in sentence.words]

        if lowercase:
            tokens = [token.lower() for token in tokens]
        if remove_diacritics:
            tokens = [unidecode.unidecode(token) for token in tokens]
        samples[i].text = tokens

    return samples


def vocabulary(samples, min_count=1):

    vocab = dict()
    for sample in samples:
        for token in sample.text:
            try:
                vocab[token] += 1
            except KeyError:
                vocab[token] = 1

    vocab = {
        key: value
        for key, value
        in vocab.items()
        if value >= min_count
    }

    return list(vocab.keys())
