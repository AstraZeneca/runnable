from pydantic import BaseModel

from magnus import get_object, put_object


class EggsModel(BaseModel):
    ham: str


class EverythingModel(BaseModel):
    spam: str
    eggs: EggsModel


def put_data_object():
    """
    Create a pydantic object that we want to pass between steps
    Store the object in the catalog for downstream steps.
    """

    data_model = EverythingModel(spam="Hello", eggs=EggsModel(ham="Yes, please!!"))

    put_object(data_model, name="everything_model")


def retrieve_object():
    """
    Retrieve the pydantic object from the catalog.
    """

    data_model = get_object("everything_model")

    print(data_model)
    ">>>spam='Hello' eggs=EggsModel(ham='Yes, please!!')"


def main():
    from magnus import Pipeline, Task

    # This step creates an object and stores it in the catalog.
    object_put = Task(
        name="create_content_in_data_folder",
        command="examples.concepts.catalog_object.put_data_object",
    )

    # This step retrieves the object from the catalog and prints its content.
    object_get = Task(
        name="retrieve_content_from_both",
        command="examples.concepts.catalog_object.retrieve_object",
        terminate_with_success=True,
    )

    object_put >> object_get

    pipeline = Pipeline(
        steps=[object_put, object_get],
        start_at=object_put,
        add_terminal_nodes=True,
    )

    # Override the default configuration file with the one that has file-system as the catalog.
    _ = pipeline.execute(configuration_file="examples/configs/fs-catalog.yaml")


if __name__ == "__main__":
    main()
