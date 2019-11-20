from load import *
from representations import *
from classifier import *

samples = load_samples('data/data.txt')
print('Data loaded')

samples = preprocess_text(samples)
vocab = vocabulary(samples)
print('Samples ready')

data = binary_data(samples, vocab)
# data = fasttext_data(samples, vocab)
# data = elmo_data(samples)
print('Representations ready')

for i in range(1, 10, 2):
    print(train(data, i/10, mlp=True))
