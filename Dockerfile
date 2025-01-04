# Python 3.8 Image without Dependecies
FROM ubuntu:24.04

LABEL maintainer="vijay.vammi@astrazeneca.com"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/0.5.12/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY . /app
WORKDIR /app

RUN uv python install && \
    uv sync --index https://artifactory.astrazeneca.net/api/pypi/pypi-virtual/simple/ --frozen --all-extras

ENV PATH="/app/.venv/bin:$PATH"
