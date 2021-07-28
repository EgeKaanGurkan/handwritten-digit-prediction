import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping


class termColors:
    WHITE = "\u001b[38;5;231m"
    BOLD = "\033[1m"
    RED = "\u001b[38;5;196m"
    RED_BACKGROUND = "\u001b[48;5;196m"
    GREEN = "\u001b[38;5;70m"
    GREEN_BACKGROUND = "\u001b[48;5;70m"
    ORANGE = "\u001b[38;5;208m"
    ORANGE_BACKGROUND = "\u001b[48;5;208m"
    GRAY = "\u001b[38;5;8m" + BOLD
    GRAY_BACKGROUND = "\u001b[48;5;8m" + BOLD
    ENDCHAR = "\u001b[0m"

if __name__ == '__main__':
    mnist_dataset = tf.keras.datasets.mnist

    # dataset_batched = mnist_dataset.batch(128).repeat(10)
    # print(dataset_batched)

    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

    print(x_train.shape)

    plt.imshow(x_train[0], cmap=plt.cm.binary)
    plt.show()

    x_train = tf.keras.utils.normalize(x_train)
    x_test = tf.keras.utils.normalize(x_test)

    print(x_train[0])

    plt.imshow(x_train[0], cmap=plt.cm.binary)
    plt.show()

    IMG_SIZE = 28

    x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    print("Training Samples dimension", x_trainr.shape)
    print("Testing Samples dimension", x_testr.shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=x_trainr.shape[1:]),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Dense(units=32),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Dense(units=10),
        tf.keras.layers.Activation("softmax")
    ])

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=3)
    model.fit(x_trainr, y_train, epochs=15, validation_split=0.3, callbacks=[early_stop])

    model.save("model")

    test_loss, test_accuracy = model.evaluate(x_testr, y_test)
    print(f"{termColors.RED_BACKGROUND}{termColors.BOLD}{termColors.WHITE} Loss on 10_000 samples: {termColors.ENDCHAR} {termColors.BOLD}{termColors.RED}" + str(test_loss) + f"{termColors.ENDCHAR}")
    print(f"{termColors.GREEN_BACKGROUND}{termColors.BOLD}{termColors.WHITE} Accuracy on 10_000 samples: {termColors.ENDCHAR} {termColors.BOLD}{termColors.GREEN}" + str(test_accuracy) + f"{termColors.ENDCHAR}")

