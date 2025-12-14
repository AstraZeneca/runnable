class RunLogExistsError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id):
        self.run_id = run_id
        message = f"Run id for {run_id} is already found in the datastore"
        super().__init__(message)


class JobLogNotFoundError(Exception):
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id):
        self.run_id = run_id
        message = f"Job for {run_id} is not found in the datastore"
        super().__init__(message)


class RunLogNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id):
        self.run_id = run_id
        message = f"Run id for {run_id} is not found in the datastore"
        super().__init__(message)


class StepLogNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id, step_name):
        self.run_id = run_id
        self.step_name = step_name
        message = f"Step log for {step_name} is not found in the datastore for Run id: {run_id}"
        super().__init__(message)


class BranchLogNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id, branch_name):
        self.run_id = run_id
        self.branch_name = branch_name
        message = f"Branch log for {branch_name} is not found in the datastore for Run id: {run_id}"
        super().__init__(message)


class NodeNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, name):
        self.name = name
        message = f"Node of name {name} is not found the graph"
        super().__init__(message)


class BranchNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, name):
        self.name = name
        message = f"Branch of name {name} is not found the graph"
        super().__init__(message)


class NodeMethodCallError(Exception):
    """
    Exception class
    """

    def __init__(self, message):
        super().__init__(message)


class TerminalNodeError(Exception):  # pragma: no cover
    def __init__(self):
        message = "Terminal Nodes do not have next node"
        super().__init__(message)


class SecretNotFoundError(Exception):  # pragma: no cover
    def __init__(self, secret_name, secret_setting):
        self.secret_name = secret_name
        self.secret_setting = secret_setting
        message = f"No secret found by name:{secret_name} in {secret_setting}"
        super().__init__(message)


class ExecutionFailedError(Exception):  # pragma: no cover
    def __init__(self, run_id: str):
        self.run_id = run_id
        message = f"Execution failed for run id: {run_id}"
        super().__init__(message)


class CommandCallError(Exception):  # pragma: no cover
    "An exception during the call of the command"
