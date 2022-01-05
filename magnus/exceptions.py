class RunLogNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id):
        super().__init__()
        self.message = f'Run id for {run_id} is not found in the datastore'


class StepLogNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id, name):
        super().__init__()
        self.message = f'Step log for {name} is not found in the datastore for Run id: {run_id}'


class BranchLogNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, run_id, name):
        super().__init__()
        self.message = f'Branch log for {name} is not found in the datastore for Run id: {run_id}'


class NodeNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, name):
        super().__init__()
        self.message = f'Node of name {name} is not found the graph'


class BranchNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, name):
        super().__init__()
        self.message = f'Branch of name {name} is not found the graph'


class UnSupportedModeError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, name, operation):
        super().__init__()
        self.message = f'Mode of type {name} is not supporterted for operation {operation}'


class SecretNotFoundError(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, secret_name, secret_setting):
        super().__init__()
        self.message = f'No secret found by name:{secret_name} in {secret_setting}'


class RunLogTypeNotInstalled(Exception):  # pragma: no cover
    """
    Exception class
    Args:
        Exception ([type]): [description]
    """

    def __init__(self, message):
        super().__init__()
        self.message = message
