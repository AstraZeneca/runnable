from typing import List

import numpy as np
from pydantic import BaseModel


class TrainParams(BaseModel):
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


class BaseLineParams(BaseModel):
    num_pixels: int = 784
    kernel_initializer: str = "normal"
    pixels_activation: str = "relu"
    classes_activation: str = "softmax"
    loss: str = "categorical_crossentropy"
    optimizer: str = "adam"

    batch_size: int = 128
    epochs: int = 15
    validation_split: float = 0.1

    metrics: List[str] = ["accuracy"]


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


def convert_to_categorically(y_train: np.ndarray, y_test: np.ndarray, num_classes: int):
    import keras

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return y_train, y_test


def build_model(train_params: TrainParams, num_classes: int):
    import keras
    from keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=train_params.input_shape),
            layers.Conv2D(
                32, train_params.kernel_size, activation=train_params.conv_activation
            ),
            layers.MaxPooling2D(pool_size=train_params.pool_size),
            layers.Conv2D(
                64,
                kernel_size=train_params.kernel_size,
                activation=train_params.conv_activation,
            ),
            layers.MaxPooling2D(pool_size=train_params.pool_size),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation=train_params.dense_activation),
        ]
    )

    print(model.summary())

    model.save("model.keras")


def build_baseline_model(baseline_params: BaseLineParams, num_classes: int):
    import keras

    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            baseline_params.num_pixels,
            input_dim=baseline_params.num_pixels,
            kernel_initializer=baseline_params.kernel_initializer,
            activation=baseline_params.pixels_activation,
        )
    )
    model.add(
        keras.layers.Dense(
            num_classes,
            kernel_initializer=baseline_params.kernel_initializer,
            activation=baseline_params.classes_activation,
        )
    )

    model.compile(
        loss=baseline_params.loss,
        optimizer=baseline_params.optimizer,
        metrics=baseline_params.metrics,
    )
    print(model.summary())

    model.save("baseline_model.keras")


def train_model(x_train: np.ndarray, y_train: np.ndarray, train_params: TrainParams):
    import keras

    model = keras.models.load_model("model.keras")
    model.compile(
        loss=train_params.loss,
        optimizer=train_params.optimizer,
        metrics=train_params.metrics,
    )

    model.fit(
        x_train,
        y_train,
        batch_size=train_params.batch_size,
        epochs=train_params.epochs,
        validation_split=train_params.validation_split,
    )

    model.save("trained_model.keras")


def train_baseline_model(
    x_train: np.ndarray, y_train: np.ndarray, train_params: BaseLineParams
):
    import keras

    model = keras.models.load_model("baseline_model.keras")
    model.compile(
        loss=train_params.loss,
        optimizer=train_params.optimizer,
        metrics=train_params.metrics,
    )

    _x_train = x_train.reshape(x_train.shape[0], train_params.num_pixels).astype(
        "float32"
    )
    # _y_train = y_train.reshape(y_train.shape[0], train_params.num_pixels).astype("float32")

    model.fit(
        _x_train,
        y_train,
        batch_size=train_params.batch_size,
        epochs=train_params.epochs,
        validation_split=train_params.validation_split,
    )

    model.save("trained_baseline_model.keras")


def evaluate_model(x_test: np.ndarray, y_test: np.ndarray):
    import keras

    trained_model = keras.models.load_model("trained_model.keras")

    score = trained_model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return score


def evaluate_baseline_model(
    x_test: np.ndarray, y_test: np.ndarray, train_params: BaseLineParams
):
    import keras

    trained_model = keras.models.load_model("trained_baseline_model.keras")

    _x_test = x_test.reshape(x_test.shape[0], train_params.num_pixels).astype("float32")

    score = trained_model.evaluate(_x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return score


def main():
    from runnable import Catalog, Parallel, Pipeline, PythonTask, metric, pickled

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

    build_baseline_model_task = PythonTask(
        function=build_baseline_model,
        name="build_baseline_model",
        catalog=Catalog(put=["baseline_model.keras"]),
    )

    train_model_task = PythonTask(
        function=train_model,
        name="train_model",
        catalog=Catalog(
            get=["model.keras"],
            put=["trained_model.keras"],
        ),
    )

    train_baseline_model_task = PythonTask(
        function=train_baseline_model,
        name="train_baseline_model",
        catalog=Catalog(
            get=["baseline_model.keras"],
            put=["trained_baseline_model.keras"],
        ),
    )

    evaluate_model_task = PythonTask(
        function=evaluate_model,
        name="evaluate_model",
        catalog=Catalog(
            get=["trained_model.keras"],
        ),
        returns=[metric("keras_score")],
        terminate_with_success=True,
    )

    evaluate_baseline_model_task = PythonTask(
        function=evaluate_baseline_model,
        name="evaluate_baseline_model",
        catalog=Catalog(
            get=["trained_baseline_model.keras"],
        ),
        returns=[metric("baseline_score")],
        terminate_with_success=True,
    )

    train_pipeline = Pipeline(
        steps=[build_model_task, train_model_task, evaluate_model_task]
    )
    baseline_train = Pipeline(
        steps=[
            build_baseline_model_task,
            train_baseline_model_task,
            evaluate_baseline_model_task,
        ]
    )

    parallel_step = Parallel(
        name="train models",
        branches={"train": train_pipeline, "baseline": baseline_train},
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[
            load_data_task,
            scale_data_task,
            convert_to_categorically_task,
            parallel_step,
        ],
    )

    pipeline.execute(parameters_file="examples/tutorials/mnist/parameters.yaml")

    return pipeline


if __name__ == "__main__":
    main()
