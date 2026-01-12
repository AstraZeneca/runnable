from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import string
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from string import Template as str_template
from typing import Any, Dict, Optional, Tuple, Union

from ruamel.yaml import YAML

import runnable.context as context
from runnable import console, defaults
from runnable.defaults import IterableParameterModel

logger = logging.getLogger(defaults.LOGGER_NAME)
logging.getLogger("stevedore").setLevel(logging.CRITICAL)


def does_file_exist(file_path: str) -> bool:
    """Check if a file exists.
    Implemented here to avoid repetition of logic.

    Args:
        file_path (str): The file path to check

    Returns:
        bool: False if it does not otherwise True
    """
    my_file = Path(file_path)
    return my_file.is_file()


def does_dir_exist(file_path: Union[str, Path]) -> bool:
    """Check if a directory exists.
    Implemented here to avoid repetition of logic.

    Args:
        file_path (str or Path): The directory path to check

    Returns:
        bool: False if the directory does not exist, True otherwise
    """
    my_file = Path(file_path)
    return my_file.is_dir()


def safe_make_dir(directory: Union[str, Path]):
    """Safely make the directory.
    Ignore if it exists and create the parents if necessary.

    Args:
        directory (str): The directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def apply_variables(
    apply_to: Dict[str, Any], variables: Dict[str, str]
) -> Dict[str, Any]:
    """Safely applies the variables to a config.

    For example: For config:
         {'a' : ${b}}, the value of ${b} is replaced by b in the variables.

    If the ${b} does not exist in the variables, it is ignored in the config.

    Args:
        apply_to (dict): The config to apply variables
        variables (dict): The variables in key, value pairs

    Raises:
        Exception: If the variables is not dict

    Returns:
        dict: A transformed dict with variables applied
    """
    if not isinstance(variables, dict):
        raise Exception("Argument Variables should be dict")

    json_d = json.dumps(apply_to)
    string_template = str_template(json_d)

    template = string_template.safe_substitute(variables)

    if "$" in template:
        logger.warning(
            "Not all variables found in the config are found in the variables"
        )

    return json.loads(template)


def get_module_and_attr_names(command: str) -> Tuple[str, str]:
    """Given a string of module.function, this functions returns the module name and func names.

    It also checks to make sure that the string is of expected 'module.func' format

    Args:
        command (str): String of format module.function_name

    Raises:
        Exception: If the string is of not format

    Returns:
        Tuple[str, str]: (module_name, function_name) extracted from the input string
    """
    mods = command.split(".")
    if len(mods) <= 1:
        raise Exception("The command should be a function to call")
    func = mods[-1]
    module = ".".join(mods[:-1])
    return module, func


def load_yaml(file_path: str, load_type: str = "safe") -> Dict[str, Any]:
    """Loads an yaml and returns the dictionary.

    Args:
        file_path (str): The path of the yamlfile
        load_type (str, optional): The load type as understood by ruamel. Defaults to 'safe'.

    Returns:
        dict: The mapping as defined in the yaml file
    """
    with open(file_path, encoding="utf-8") as f:
        yaml = YAML(typ=load_type, pure=True)
        yaml_config = yaml.load(f)
    return yaml_config


def is_a_git_repo() -> bool:
    """Does a git command to see if the project is git versioned.

    Returns:
        bool: True if it is git versioned, False otherwise
    """
    command = "git rev-parse --is-inside-work-tree"
    try:
        subprocess.check_output(command.split()).strip().decode("utf-8")
        logger.info("Found the code to be git versioned")
        return True
    except BaseException:  # pylint: disable=W0702
        console.print("Not a git repo", style="bold red")

    return False


def get_current_code_commit() -> Union[str, None]:
    """Gets the git sha id if the project is version controlled.

    Returns:
        Union[str, None]: SHA ID if the code is versioned, None otherwise
    """
    command = "git rev-parse HEAD"
    if not is_a_git_repo():
        return None
    try:
        label = subprocess.check_output(command.split()).strip().decode("utf-8")
        logger.info("Found the git commit to be: %s", label)
        return label
    except BaseException:  # pylint: disable=W0702
        console.print("Not a git repo, error getting hash", style="bold red")
        raise


def is_git_clean() -> Tuple[bool, Union[None, str]]:
    """Checks if the git tree is clean and there are no modified tracked files.

    Returns:
        Tuple[bool, Union[None, str]]: None if its clean, comma-seperated file names if it is changed
    """
    command = "git diff --name-only"
    if not is_a_git_repo():
        return False, None
    try:
        label = subprocess.check_output(command.split()).strip().decode("utf-8")
        if not label:
            return True, None
        return False, label
    except BaseException:  # pylint: disable=W0702
        console.print("Not a git repo, not clean", style="bold red")

    return False, None


def get_git_remote() -> Union[str, None]:
    """Gets the remote URL of git.

    Returns:
        Union[str, None]: Remote URL if the code is version controlled, None otherwise
    """
    command = "git config --get remote.origin.url"
    if not is_a_git_repo():
        return None
    try:
        label = subprocess.check_output(command.split()).strip().decode("utf-8")
        logger.info("Found the git remote to be: %s", label)
        return label
    except BaseException:  # pylint: disable=W0702
        console.print("Not a git repo, no remote", style="bold red")
        raise


def get_local_docker_image_id(image_name: str) -> str:
    """If we are running in local settings, return the docker image id.

    Args:
        image_name (str): The image name we need the digest for

    Returns:
        str: The docker image digest
    """
    try:
        import docker

        client = docker.from_env()
        image = client.images.get(image_name)
        return image.attrs["Id"]
    except ImportError:  # pragma: no cover
        logger.warning(
            "Did not find docker installed, some functionality might be affected"
        )
    except BaseException:
        logger.exception(f"Could not find the image by name {image_name}")

    return ""


def get_git_code_identity():
    """Returns a code identity object for version controlled code.

    Args:
        run_log_store (runnable.datastore.BaseRunLogStore): The run log store used in this process

    Returns:
        runnable.datastore.CodeIdentity: The code identity used by the run log store.
    """
    current_context = context.get_run_context()
    if current_context is None:
        raise RuntimeError("No run context available")
    code_identity = current_context.run_log_store.create_code_identity()
    try:
        code_identity.code_identifier = get_current_code_commit()
        code_identity.code_identifier_type = "git"
        code_identity.code_identifier_dependable, changed = is_git_clean()
        code_identity.code_identifier_url = get_git_remote()
        if changed:
            code_identity.code_identifier_message = "changes found in " + ", ".join(
                changed.split("\n")
            )
    except BaseException:
        logger.exception("Git code versioning problems")

    return code_identity


def remove_prefix(text: str, prefix: str) -> str:
    """Removes a prefix if one is present in the input text.

    Args:
        text (str): The input text to remove the prefix from
        prefix (str): The prefix that has to be removed

    Returns:
        str: The original string if no prefix is found, or the right prefix chomped string if present
    """
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever is given


def diff_dict(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given two dicts d1 and d2, return a new dict that has upsert items from d1.

    Args:
        d1 (reference): The reference dict.
        d2 (compare): Any new or modified items compared to d1 would be returned back

    Returns:
        dict: Any new or modified items in d2 in comparison to d1 would be sent back
    """
    diff = {}

    for k2, v2 in d2.items():
        if k2 in d1 and d1[k2] != v2:
            diff[k2] = v2
            continue
        diff[k2] = v2

    return diff


def get_data_hash(file_name: str) -> str:
    """Returns the hash of the data file.

    For small files (<1GB): Returns full SHA256 hash
    For large files (>=1GB): Returns fingerprint hash of first chunk + last chunk + file size

    Args:
        file_name (str): The file name to generate the hash for

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read due to permissions
        OSError: If there are other I/O errors

    Returns:
        str: The SHA256 hash or fingerprint of the file contents
    """
    start_time = time.time()

    try:
        file_path = Path(file_name)
        file_size = file_path.stat().st_size

        # Use appropriate algorithm based on file size
        if file_size < defaults.LARGE_FILE_THRESHOLD_BYTES:
            result = _compute_full_file_hash(file_name)
            logger.debug(
                f"Full hash computed for {file_name} ({file_size} bytes) in {time.time() - start_time:.3f}s"
            )
        else:
            result = _compute_large_file_fingerprint(file_name, file_size)
            logger.info(
                f"Fingerprint hash computed for {file_name} ({file_size} bytes) in {time.time() - start_time:.3f}s"
            )

        return result
    except FileNotFoundError:
        logger.error(f"File not found: {file_name}")
        raise
    except PermissionError:
        logger.error(f"Permission denied accessing file: {file_name}")
        raise
    except OSError as e:
        logger.error(f"I/O error accessing file {file_name}: {e}")
        raise


def _compute_full_file_hash(file_name: str) -> str:
    """Compute SHA256 hash of entire file using streaming approach."""
    with open(file_name, "rb") as f:
        file_hash = hashlib.sha256()
        for chunk in iter(lambda: f.read(4096), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def _compute_large_file_fingerprint(file_name: str, file_size: int) -> str:
    """Compute fingerprint hash for large files using first/last chunks + metadata."""
    with open(file_name, "rb") as f:
        file_hash = hashlib.sha256()

        # Include file size in hash for uniqueness
        file_hash.update(str(file_size).encode())

        # Read first chunk
        first_chunk = f.read(defaults.HASH_CHUNK_SIZE)
        file_hash.update(first_chunk)

        # Read last chunk if file is large enough and different from first chunk
        if file_size > defaults.HASH_CHUNK_SIZE:
            f.seek(-min(defaults.HASH_CHUNK_SIZE, file_size - len(first_chunk)), 2)
            last_chunk = f.read(defaults.HASH_CHUNK_SIZE)
            file_hash.update(last_chunk)

    return file_hash.hexdigest()


def json_to_ordered_dict(json_str: str) -> OrderedDict:
    """Decode a JSON str into OrderedDict.

    Args:
        json_str ([str]): The JSON string to decode

    Returns:
        [OrderedDict]: The decoded OrderedDict
    """
    if json_str and json_str != "{}":
        return json.loads(json_str, object_pairs_hook=OrderedDict)

    return OrderedDict()


def gather_variables() -> Dict[str, str]:
    """Gather all the environment variables used by runnable. All the variables start with runnable_VAR_.

    Returns:
        dict: All the environment variables present in the environment.
    """
    variables = {}

    for env_var, value in os.environ.items():
        if env_var.startswith(defaults.VARIABLE_PREFIX):
            key = remove_prefix(env_var, defaults.VARIABLE_PREFIX)
            variables[key] = value

    return variables


def make_log_file_name(
    name: str,
    iter_variable: Optional[IterableParameterModel] = None,
) -> str:
    random_tag = "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
    log_file_name = name

    if iter_variable and iter_variable.map_variable:
        for _, value in iter_variable.map_variable.items():
            log_file_name += "_" + str(value)

    log_file_name += "_" + random_tag
    log_file_name = "".join(x for x in log_file_name if x.isalnum()) + ".execution.log"

    return log_file_name
