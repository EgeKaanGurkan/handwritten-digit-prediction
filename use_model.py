import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    model = tf.keras.models.load_model("model")

    model.summary()

    mnist_dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

    IMG_SIZE = 28

    x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    predictions = model.predict([x_testr])

    img_index = randint(0, 200)
    print(f"Image index: {img_index}")

    plt.imshow(x_testr[img_index], cmap=plt.cm.binary)
    plt.show()

    print(np.argmax(predictions[img_index]))

    figure = plt.figure()

    rows = 5
    columns = 5

    for i in range(0, 25):
        img_index = randint(0, 200)

        figure.add_subplot(rows, columns, i + 1)

        plt.margins(1)

        plt.imshow(x_testr[img_index], cmap=plt.cm.binary)
        plt.title(np.argmax(predictions[img_index]))

    plt.show()
