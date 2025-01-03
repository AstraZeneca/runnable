class RunLogExistsError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id):
        super().__init__()
        self.message = f"Run id for {run_id} is already found in the datastore"


class JobLogNotFoundError(Exception):
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id):
        super().__init__()
        self.message = f"Job for {run_id} is not found in the datastore"


class RunLogNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id):
        super().__init__()
        self.message = f"Run id for {run_id} is not found in the datastore"


class StepLogNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id, name):
        super().__init__()
        self.message = (
            f"Step log for {name} is not found in the datastore for Run id: {run_id}"
        )


class BranchLogNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id, name):
        super().__init__()
        self.message = (
            f"Branch log for {name} is not found in the datastore for Run id: {run_id}"
        )


class NodeNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, name):
        super().__init__()
        self.message = f"Node of name {name} is not found the graph"


class BranchNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, name):
        super().__init__()
        self.message = f"Branch of name {name} is not found the graph"


class NodeMethodCallError(Exception):
    """
    Exception class
    """

    def __init__(self, message):
        super().__init__()
        self.message = message


class TerminalNodeError(Exception):  # pragma: no cover
    def __init__(self):
        super().__init__()
        self.message = "Terminal Nodes do not have next node"


class SecretNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, secret_name, secret_setting):
        super().__init__()
        self.message = f"No secret found by name:{secret_name} in {secret_setting}"


class ExecutionFailedError(Exception):  # pragma: no cover
    def __init__(self, run_id: str):
        super().__init__()
        self.message = f"Execution failed for run id: {run_id}"


class CommandCallError(Exception):  # pragma: no cover
    "An exception during the call of the command"
