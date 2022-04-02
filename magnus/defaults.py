NAME = 'magnus'

# CLI settings
LOG_LEVEL = 'WARNING'

# Interaction settings
TRACK_PREFIX = 'MAGNUS_TRACK_'
PARAMETER_PREFIX = 'MAGNUS_PRM_'

# STATUS progression
# For Branch, CREATED -> PROCESSING -> SUCCESS OR FAIL
# For a step, CREATED -> TRIGGERED ->  PROCESSING -> SUCCESS OR FAIL
CREATED = 'CREATED'
PROCESSING = 'PROCESSING'
SUCCESS = 'SUCCESS'
FAIL = 'FAIL'
TRIGGERED = 'TRIGGERED'

# Node and Command settings
COMMAND_TYPE = 'python'
NODE_SPEC_FILE = 'node_spec.yaml'
COMMAND_FRIENDLY_CHARACTER = '%'

# Default services
DEFAULT_EXECUTOR = {
    'type': 'local'
}
DEFAULT_RUN_LOG_STORE = {
    'type': 'buffered'
}
DEFAULT_CATALOG = {
    'type': 'file-system'
}
DEFAULT_SECRETS = {
    'type': 'do-nothing'
}

# Map state
MAP_PLACEHOLDER = 'map_variable_placeholder'

# Dag node
DAG_BRANCH_NAME = 'dag'

# RUN settings
RANDOM_RUN_ID_LEN = 6
MAX_TIME = 86400  # 1 day in seconds

# User extensions
USER_CONFIG_FILE = 'magnus-config.yaml'

# Executor settings
ENABLE_PARALLEL = False

# RUN log store settings
LOG_LOCATION_FOLDER = '.run_log_store'

# Dag node
DAG_BRANCH_NAME = 'dag'

# Data catalog settings
CATALOG_LOCATION_FOLDER = '.catalog'
COMPUTE_DATA_FOLDER = 'data'

# Secrets settings
DOTENV_FILE_LOCATION = '.env'

# AWS settings
AWS_REGION = 'eu-west-1'

# Docker settings
DOCKERFILE_NAME = 'Dockerfile'
DOCKERFILE_CONTENT = r"""# Python 3.6 Image without Dependecies
FROM python:3.7

LABEL maintainer="vijay.vammi@astrazeneca.com"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

${INSTALL_STYLE}

ENV VIRTUAL_ENV=/opt/venv
RUN python -m virtualenv --python=/usr/local/bin/python $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

${COPY_CONTENT}
WORKDIR /app

${INSTALL_REQUIREMENTS}
"""
GIT_ARCHIVE_NAME = 'git_tracked'
LEN_SHA_FOR_TAG = 8
