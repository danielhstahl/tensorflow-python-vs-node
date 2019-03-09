from __future__ import absolute_import, division, print_function

# pip install tensorflow==2.0.0-alpha0

import tensorflow as tf

mnist = tf.keras.datasets.mnist

def convert_to_double(data):
    num_colors=255.0
    return data/num_colors


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = convert_to_double(x_train), convert_to_double(x_test)
    return (x_train, y_train), (x_test, y_test)

def train_model(x_train, y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=5)
    return model


(x_train, y_train), (x_test, y_test)=load_data()

model=train_model(x_train, y_train)

model.evaluate(x_test, y_test)