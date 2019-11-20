results = [
    [0.9101124, 0.90225565, 0.88235295, 0.84006464, 0.80778897],
    [0.8988764, 0.90225565, 0.89366513, 0.85783523, 0.8040201],
    [0.92134833, 0.9135338, 0.89819, 0.84491116, 0.7663317],
    [0.9325843, 0.9285714, 0.92081445, 0.89660746, 0.8341709],
    [0.9325843, 0.9548872, 0.9276018, 0.9095315, 0.8844221],
    [0.92134833, 0.9511278, 0.91855204, 0.9095315, 0.88065326]
]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
axes[0].plot([10, 30, 50, 70, 90], results[0])
axes[1].plot([10, 30, 50, 70, 90], results[1])
axes[0].plot([10, 30, 50, 70, 90], results[2])
axes[1].plot([10, 30, 50, 70, 90], results[3])
axes[0].plot([10, 30, 50, 70, 90], results[4])
axes[1].plot([10, 30, 50, 70, 90], results[5])

axes[0].set_ylim([0.5, 1])
axes[0].set_xlim([0, 100])
axes[0].set_xticks([10, 30, 50, 70, 90])
axes[0].set_ylabel('accuracy')
axes[0].set_xlabel('validation split [%]')
axes[1].set_xlabel('validation split [%]')
axes[0].set_title('logistic regression')
axes[1].set_title('multilayer perceptron')
axes[0].legend(['binary', 'fasttext', 'elmo'])

plt.show()