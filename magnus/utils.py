import subprocess
import logging
from datetime import datetime
import uuid
from pathlib import Path
from collections import OrderedDict
import os
import json
import hashlib
from string import Template as str_template
from inspect import signature
from typing import Tuple, Union, Callable

from ruamel.yaml import YAML

from magnus import defaults


logger = logging.getLogger(defaults.NAME)


def does_file_exist(file_path: str) -> bool:
    """
    Check if a file exists.
    Implemeted here to avoid repetition of logic.

    Args:
        file_path (str): The file path to check

    Returns:
        bool: False if it does not otherwise True
    """
    my_file = Path(file_path)
    return my_file.is_file()


def does_dir_exist(file_path: str) -> bool:
    """
    Check if a directory exists.
    Implemented here to avoid repetition of logic

    Args:
        file_path (str): The directory path to check

    Returns:
        bool: False if the directory does not exist, True otherwise
    """
    my_file = Path(file_path)
    return my_file.is_dir()


def safe_make_dir(directory: str):
    """
    Safely make the directory.
    Ignore if it exists and create the parents if necessary.

    Args:
        directory (str): The directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def validate_run_id(run_id: str):
    """
    A valid run_id should be of len (2) or less if split against _

    Args:
        run_id (str): The run_id to check

    Raises:
        Exception: If the run_id is not valid
    """
    if len(run_id.split('_')) > 2:
        raise Exception('Run id is of format, <User Identifiable>_<Random Tag>, \
                        please do not use underscores in your run_id')


def generate_run_id(run_id: str = None) -> str:
    """
    Generate a new run_id.

    If the input run_id is none, we create one based on time stamp.
    If a run_id is given, validate the run_id (len should be less than or equal to 2).

    A FULL run_id = some_random-characters
    If the run_id is FULL specification, we ignore the random part of it and add some random bits
    If the run_id is not full, we add random bits to the end of it to make it FULL

    Args:
        run_id (str, optional): Input Run ID. Defaults to None

    Returns:
        str: A generated run_id
    """
    # If we are not provided with a run_id, generate one
    if not run_id:
        run_id = datetime.now().strftime('%Y%m%d%H%M%S')

    validate_run_id(run_id)
    # If we are, just add a random tag to the run_id
    run_id = run_id.split('_')[0]

    random_tag = str(uuid.uuid4())[:defaults.RANDOM_RUN_ID_LEN]
    return run_id + '_' + random_tag


def apply_variables(apply_to: dict, variables: dict) -> dict:
    """
    Safely applies the variables to a config.

    For example: For config:
         {'a' : ${b}}, the value of ${b} is replaced by b in the variables.

    If the ${b} does not exist in the variables, it is ignored in the config.

    Args:
        apply_to (dict): The config to apply variables
        variables (dict): The variables in key, value pairs

    Raises:
        Exception: If the variables is not dict

    Returns:
        dict: A trasformed dict with variables applied
    """
    if type(variables) != dict:
        raise Exception('Argument Variables should be dict')

    json_d = json.dumps(apply_to)
    transformed = str_template(json_d).safe_substitute(**variables)
    return json.loads(transformed)


def get_module_and_func_names(command: str) -> Tuple[str, str]:
    """
    Given a string of module.function, this functions returns the module name and func names.

    It also checks to make sure that the string is of expected 'module.func' format

    Args:
        command (str): String of format module.function_name

    Raises:
        Exception: If the string is of not format

    Returns:
        Tuple[str, str]: (module_name, function_name) extracted from the input string
    """
    mods = command.split('.')
    if len(mods) <= 1:
        raise Exception('The command should be a function to call')
    func = mods[-1]
    module = '.'.join(mods[:-1])
    return module, func


def get_dag_hash(dag: dict) -> str:
    """
    Generates the hash of the dag definition

    Args:
        dag (dict): The dictionary object containing the dag definition

    Returns:
        str: The hash of the dag definition
    """
    dag_str = json.dumps(dag, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(dag_str.encode('utf-8')).hexdigest()


def load_yaml(file_path: str, load_type: str = 'safe') -> dict:
    """
    Loads an yaml and returns the dictionary

    Args:
        file_path (str): The path of the yamlfile
        load_type (str, optional): The load type as understood by ruamel. Defaults to 'safe'.

    Returns:
        dict: The mapping as defined in the yaml file
    """
    with open(file_path, encoding='utf-8') as f:
        yaml = YAML(typ=load_type, pure=True)
        yaml_config = yaml.load(f)
    return yaml_config


def is_a_git_repo() -> bool:
    """
    Does a git command to see if the project is git versioned

    Returns:
        bool: True if it is git versioned, False otherwise
    """
    command = 'git rev-parse --is-inside-work-tree'
    try:
        subprocess.check_output(
            command.split()).strip().decode('utf-8')
        logger.info('Found the code to be git versioned')
        return True
    except:  # pylint: disable=W0702
        logger.error('No git repo found, unsafe hash')
        raise

    return False


def get_current_code_commit() -> Union[str, None]:
    """
    Gets the git sha id if the project is version controlled

    Returns:
        Union[str, None]: SHA ID if the code is versioned, None otherwise
    """
    command = 'git rev-parse HEAD'
    if not is_a_git_repo():
        return None
    try:
        label = subprocess.check_output(
            command.split()).strip().decode('utf-8')
        logger.info("Found the git commit to be: %s", label)
        return label
    except:  # pylint: disable=W0702
        logger.exception('Error getting git hash')
        raise

    return None


def is_git_clean() -> Tuple[bool, Union[None, str]]:
    """
    Checks if the git tree is clean and there are no modified tracked files

    Returns:
        Tuple[bool, Union[None, str]]: None if its clean, comma-seperated file names if it is changed
    """
    command = 'git diff --name-only'
    if not is_a_git_repo():
        return False, None
    try:
        label = subprocess.check_output(
            command.split()).strip().decode('utf-8')
        if not label:
            return True, None
        return False, label
    except:  # pylint: disable=W0702
        logger.exception('Error checking if the code is git clean')
    return False, None


def get_git_remote() -> Union[str, None]:
    """
    Gets the remote URL of git

    Returns:
        Union[str, None]: Remote URL if the code is version controlled, None otherwise
    """
    command = 'git config --get remote.origin.url'
    if not is_a_git_repo():
        return None
    try:
        label = subprocess.check_output(
            command.split()).strip().decode('utf-8')
        logger.info('Found the git remote to be: %s', label)
        return label
    except:  # pylint: disable=W0702
        logger.exception('Error getting git remote')

    return None


def get_local_docker_image_id(image_name: str) -> str:
    """
    If we are running in local settings, return the docker image id

    Args:
        image_name (str): The image name we need the digest for

    Returns:
        str: The docker image digest
    """
    try:
        import docker
        client = docker.from_env()
        image = client.images.get(image_name)
        return image.attrs['Id']
    except ImportError:  # pragma: no cover
        logger.warning('Did not find docker installed, some functionality might be affected')
    except:
        logger.exception(f'Could not find the image by name {image_name}')

    return None


def get_git_code_identity(run_log_store):
    # TODO: Bug in this implementation
    """
    Returns a code identity object for version controlled code.

    Args:
        run_log_store (magnus.datastore.BaseRunLogStore): The run log store used in this process

    Returns:
        magnus.datastore.CodeIdentity: The code identiy used by the run log store.
    """
    try:
        code_identity = run_log_store.create_code_identity()
        code_identity.code_identifier = get_current_code_commit()
        code_identity.code_identifier_type = 'git'
        code_identity.code_identifer_dependable, changed = is_git_clean()
        code_identity.code_identifier_url = get_git_remote()
        if changed:
            code_identity.code_identifier_message = 'changes found in ' + ', '.join(changed.split('\n'))
        return code_identity
    except:
        logger.exception("Git code versioning problems")
        return None


def remove_prefix(text: str, prefix: str) -> str:
    """
    Removes a prefix if one is present in the input text

    Args:
        text (str): The input text to remove the prefix from
        prefix (str): The prefix that has to be removed

    Returns:
        str: The original string if no prefix is found, or the right prefix chomped string if present
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever is given


def get_tracked_data() -> dict:
    """
    Scans the environment variables to find any user tracked variables that have a prefix MAGNUS_TRACK_
    Removes the environment varaible to prevent any clashes in the future steps

    Returns:
        dict: A dictionary of user tracked data
    """
    tracked_data = {}
    for env_var, value in os.environ.items():
        if env_var.startswith(defaults.TRACK_PREFIX):
            key = remove_prefix(env_var, defaults.TRACK_PREFIX)
            tracked_data[key.lower()] = json.loads(value)
            del os.environ[env_var]
    return tracked_data


def get_user_set_parameters(remove: bool = False) -> dict:
    """
    Scans the environment variables for any user returned parameters that have a prefix MAGNUS_PRM_

    Args:
        remove (bool, optional): Flag to remove the parameter if needed. Defaults to False.

    Returns:
        dict: The dictionary of found user returned parameters
    """
    parameters = {}
    for env_var, value in os.environ.items():
        if env_var.startswith(defaults.PARAMETER_PREFIX):
            key = remove_prefix(env_var, defaults.PARAMETER_PREFIX)
            parameters[key.lower()] = json.loads(value)
            if remove:
                del os.environ[env_var]
    return parameters


def hash_bytestr_iter(bytesiter, hasher, ashexstr=True):  # pylint: disable=C0116
    for block in bytesiter:  # pragma: no cover
        hasher.update(block)
    return hasher.hexdigest() if ashexstr else hasher.digest()  # pragma: no cover


def file_as_blockiter(afile, blocksize=65536):  # pylint: disable=C0116
    with afile:  # pragma: no cover
        block = afile.read(blocksize)
        while len(block) > 0:
            yield block
            block = afile.read(blocksize)


def get_data_hash(file_name: str):
    """
    Returns the hash of the data file

    Args:
        file_name (str): The file name to generated the hash

    Returns:
        str: The SHA ID of the file contents
    """
    # https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
    return hash_bytestr_iter(file_as_blockiter(open(file_name, 'rb')), hashlib.sha256())  # pragma: no cover


def filter_arguments_for_func(func: Callable, parameters: dict, map_variable: dict) -> dict:
    """
    Inspects the function to be called as part of the pipeline to find the arguments of the function.
    Matches the function arguments to the parameters available either by command line or by up stream steps.

    Args:
        func (Callable): The function to inspect
        parameters (dict): The parameters available for the run

    Returns:
        dict: The parameters matching the function signature
    """
    sign = signature(func)
    filtered_parameters = {}
    for param, value in parameters.items():
        if param in sign.parameters:
            filtered_parameters[param] = value

    if map_variable:
        for iterate_as, value in map_variable.items():
            if iterate_as in sign.parameters:
                filtered_parameters[iterate_as] = value

    return filtered_parameters


def get_node_execution_command(executor, node, map_variable=None, over_write_run_id=None) -> str:
    """
    A utility function to standardise execution call to a node via command line.

    Args:
        executor (object): The executor class.
        node (object): The Node to execute
        map_variable (str, optional): If the node belongs to a map step. Defaults to None.

    Returns:
        str: The execution command to run a node via command line.
    """
    run_id = executor.run_id

    if over_write_run_id:
        run_id = over_write_run_id

    action = (f'magnus execute_single_node {run_id} '
              f'{node.command_friendly_name()} --file {executor.pipeline_file}'
              )

    if executor.variables_file:
        action = action + f' --var-file {executor.variables_file}'

    if map_variable:
        action = action + f' --map-variable {json.dumps(map_variable)}'

    return action


def get_service_base_class(service_type: str):
    """
    Return the BaseClass for a service type

    Args:
        service_type (str): Should be one of executor, run_log_store, catalog or secrets

    Raises:
        Exception: If the service type is not one of the accepted values

    Object:
        [type]: A base class of the service
    """
    if service_type == 'executor':
        from magnus import executor  # pylint: disable=C0415
        return executor.BaseExecutor

    if service_type == 'run_log_store':
        from magnus import datastore  # pylint: disable=C0415
        return datastore.BaseRunLogStore

    if service_type == 'catalog':
        from magnus import catalog  # pylint: disable=C0415
        return catalog.BaseCatalog

    if service_type == 'secrets':
        from magnus import secrets  # pylint: disable=C0415
        return secrets.BaseSecrets

    raise Exception('Service type is not recognised')


def get_subclasses(cls):
    """
    Give a class, return all its subclasses.

    This method would return even in case of heirarchial sub classes.

    Yields:
        obect: The subclass
    """
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


def get_provider_by_name_and_type(service_type: str, service_details: dict):
    """Given a service type, one of executor, run_log_store, catalog, secrets and the config
    return the exact child class implementing the service.

    Args:
        service_type (str): One of executor, run_log_store, catalog, secrets
        service_details (dict): The config used to instantiate the service.

    Raises:
        Exception: If the service by that name does not exist

    Returns:
        object: A service object
    """
    base_class = get_service_base_class(service_type=service_type)

    service_name = service_details['type']
    service_config = {}
    if 'config' in service_details:
        service_config = service_details.get('config', {})

    logger.info(f'Trying to get a service of {service_type} of the name {service_name} with config: {service_config}')
    for sub_class in get_subclasses(base_class):
        if service_name == sub_class.service_name:
            return sub_class(service_config)

    raise Exception(f'Could not find the service of type: {service_type} with config: {service_details}')


def get_duration_between_datetime_strings(start_time: str, end_time: str) -> str:
    """Given two datetime strings, compute the duration between them

    Args:
        start_time (str): ISO format datetime string
        end_time (str): ISO format datetime string
    Returns:
        The duration between the time in string format
    """
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)

    return str(end - start)


def get_run_config(executor: object) -> dict:
    """
    Given an executor with assigned services, return the run_config

    Args:
        executor (object): The executor with all the services assigned.

    Returns:
        dict: The run_config.
    """
    run_config = {}

    run_config['executor'] = {'type': executor.service_name,
                              'config': executor.config}

    run_config['run_log_store'] = {'type': executor.run_log_store.service_name,
                                   'config': executor.run_log_store.config}

    run_config['catalog'] = {'type': executor.catalog_handler.service_name,
                             'config': executor.catalog_handler.config}

    run_config['secrets'] = {'type': executor.secrets_handler.service_name,
                             'config': executor.secrets_handler.config}

    return run_config


def json_to_ordered_dict(json_str: str) -> OrderedDict:
    """
    Decode a JSON str into OrderedDict

    Args:
        json_str ([str]): The JSON string to decode

    Returns:
        [OrderedDict]: The decoded OrderedDict
    """

    if json_str and json_str != '{}':
        return json.loads(json_str, object_pairs_hook=OrderedDict)

    return OrderedDict()
