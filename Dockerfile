# Python 3.8 Image without Dependecies
FROM python:3.10.16-slim

LABEL maintainer="vijay.vammi@astrazeneca.com"

ENV http_proxy=http://azpse.astrazeneca.net:9480
ENV https_proxy=http://azpse.astrazeneca.net:9480
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

ENV http_proxy=
ENV https_proxy=

RUN uv sync --verbose  --frozen --extra examples --extra notebook


ENV PATH="/app/.venv/bin:$PATH"
