from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


def train(data, split, mlp=False):

    if mlp:
        model = Sequential([
            Dense(
                units=200,
                activation='relu',
            ),
            Dense(
                units=2,
                activation='softmax',
            )
        ])
    else:
        model = Sequential([
            Dense(
                units=2,
                activation='softmax',
            )
        ])

    model.compile(
        optimizer=SGD(0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        x=data.x,
        y=data.y,
        batch_size=10,
        epochs=200,
        validation_split=split,
        verbose=0,
    )

    return max(history.history['val_acc'])
