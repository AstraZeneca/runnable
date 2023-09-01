import logging
from string import Template
from typing import Union

from magnus import defaults, utils

logger = logging.getLogger(defaults.LOGGER_NAME)
try:
    import docker
except ImportError:
    logger.info("docker was not installed, docker functionality will not work")


def generate_docker_file(style: str = "poetry", git_tracked: bool = True):
    """
    Generates a Dockerfile based on the given style.

    Args:
        style (str, optional): One of python dependency style. Defaults to "poetry".
        git_tracked (bool, optional): Allow only git tracked or not. Defaults to True.

    Raises:
        Exception: If the current directory is not a git repository.
    """
    install_style, install_requirements, copy_content = None, None, None

    if style == "poetry":
        install_style = "RUN pip install poetry"
        install_requirements = "RUN poetry install"
        logger.info("Using poetry style for requirements")
    elif style == "pipenv":
        install_style = "RUN pip install pipenv"
        install_requirements = "RUN pipenv install"
        logger.info("Using pipenv style for requirements")
    else:
        install_requirements = "RUN pip install -r requirements.txt"
        logger.info("Trying requirements.txt, if one exists")

    copy_content = "COPY . /app"
    if git_tracked:
        if not utils.is_a_git_repo():
            msg = (
                "The current project is not git versioned. Disable only git tracked to create the image. "
                "Be aware over-riding this can cause leak of sensitive data if you are not careful."
            )
            raise Exception(msg)
        copy_content = "ADD git_tracked.tar.gz /app"
        utils.archive_git_tracked(defaults.GIT_ARCHIVE_NAME)

    dockerfile_content = Template(defaults.DOCKERFILE_CONTENT).safe_substitute(
        {
            "INSTALL_STYLE": install_style,
            "INSTALL_REQUIREMENTS": install_requirements,
            "COPY_CONTENT": copy_content,
        }
    )
    with open(defaults.DOCKERFILE_NAME, "w", encoding="utf-8") as fw:
        fw.write(dockerfile_content)


def build_docker(
    image_name: str,
    docker_file: Union[str, None],
    style: str,
    tag: str,
    commit_tag: bool,
    dry_run: bool = False,
    git_tracked: bool = True,
):
    """
    Builds the docker image.

    Args:
        image_name (str): The name of the docker image.
        docker_file (Union[str, None]): The name of the dockerfile, if present or None
        style (str): The style of python dependency.
        tag (str): The tag of the docker image.
        commit_tag (bool): Use the commit id for the docker image tag.
        dry_run (bool, optional): Do not run just show what would be done. Defaults to False.
        git_tracked (bool, optional): Only the git tracked files or not. Defaults to True.

    Raises:
        Exception: If the current directory is not a git repository and asked for a git tag as docker image.
    """
    if commit_tag:
        if not utils.is_a_git_repo():
            msg = "The current project is not git versioned, cannot use commit tag option when building image"
            raise Exception(msg)

        tag = utils.get_current_code_commit()[: defaults.LEN_SHA_FOR_TAG]  # type: ignore

    if not docker_file:
        generate_docker_file(style, git_tracked=git_tracked)
        docker_file = defaults.DOCKERFILE_NAME

    if dry_run:
        return

    docker_client = docker.from_env()
    docker_client.images.build(path=".", dockerfile=docker_file, tag=f"{image_name}:{tag}", quiet=False)
