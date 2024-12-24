from typing import List

import numpy as np

num_classes: int = 10
input_shape: tuple = (28, 28, 1)

kernel_size: tuple = (3, 3)
pool_size: tuple = (2, 2)

conv_activation: str = "relu"
dense_activation: str = "softmax"

loss: str = "categorical_crossentropy"
optimizer: str = "adam"
metrics: List[str] = ["accuracy"]

batch_size: int = 128
epochs: int = 15
validation_split: float = 0.1


def load_data():
    import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    return x_train, y_train, x_test, y_test


def scale_data(x_train: np.ndarray, x_test: np.ndarray):
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_test


def convert_to_categorically(y_train: np.ndarray, y_test: np.ndarray):
    import keras

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return y_train, y_test


def build_model():
    import keras
    from keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size, activation=conv_activation),
            layers.MaxPooling2D(pool_size=pool_size),
            layers.Conv2D(64, kernel_size=kernel_size, activation=conv_activation),
            layers.MaxPooling2D(pool_size=pool_size),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation=dense_activation),
        ]
    )

    print(model.summary())

    model.save("model.keras")


def train_model(x_train: np.ndarray, y_train: np.ndarray):
    import keras

    model = keras.models.load_model("model.keras")
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
    )

    model.save("trained_model.keras")


def evaluate_model(x_test: np.ndarray, y_test: np.ndarray):
    import keras

    trained_model = keras.models.load_model("trained_model.keras")

    score = trained_model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return score


def main():
    from runnable import Catalog, Pipeline, PythonTask, metric, pickled

    # x_train, y_train, x_test, y_test
    load_data_task = PythonTask(
        function=load_data,
        name="load_data",
        returns=[
            pickled("x_train"),
            pickled("y_train"),
            pickled("x_test"),
            pickled("y_test"),
        ],
    )

    # def scale_data(x_train: np.ndarray, x_test: np.ndarray)
    scale_data_task = PythonTask(
        function=scale_data,
        name="scale_data",
        returns=[pickled("x_train"), pickled("x_test")],
    )

    convert_to_categorically_task = PythonTask(
        function=convert_to_categorically,
        name="convert_to_categorically",
        returns=[pickled("y_train"), pickled("y_test")],
    )

    build_model_task = PythonTask(
        function=build_model,
        name="build_model",
        catalog=Catalog(put=["model.keras"]),
    )

    train_model_task = PythonTask(
        function=train_model,
        name="train_model",
        catalog=Catalog(
            get=["model.keras"],
            put=["trained_model.keras"],
        ),
    )

    evaluate_model_task = PythonTask(
        function=evaluate_model,
        name="evaluate_model",
        catalog=Catalog(
            get=["trained_model.keras"],
        ),
        returns=[metric("score")],
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[
            load_data_task,
            scale_data_task,
            convert_to_categorically_task,
            build_model_task,
            train_model_task,
            evaluate_model_task,
        ],
    )

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
