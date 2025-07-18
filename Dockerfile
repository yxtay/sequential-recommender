# hadolint global ignore=DL3008
# kics-scan disable=965a08d7-ef86-4f14-8792-4a3b2098937e,451d79dc-0588-476a-ad03-3c7f0320abb3
FROM ghcr.io/astral-sh/uv:0.8.0@sha256:5778d479c0fd7995fedd44614570f38a9d849256851f2786c451c220d7bd8ccd AS uv

##
# base
##
FROM debian:stable-slim@sha256:7e0b7fe7c6d695d615eabaea8d19adf592a6a9ff3dbd5206d3e31139b9afdfa7 AS base

# set up user
ARG USER=user
ARG UID=1000
RUN useradd --create-home --shell /bin/false --uid ${UID} ${USER}

# set up environment
ARG APP_HOME=/work/app
ARG VIRTUAL_ENV=${APP_HOME}/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    UV_LOCKED=1 \
    UV_NO_SYNC=1 \
    UV_PYTHON_DOWNLOADS=manual \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=${VIRTUAL_ENV}

WORKDIR ${APP_HOME}

##
# dev
##
FROM base AS dev

ARG DEBIAN_FRONTEND=noninteractive
COPY <<-EOF /etc/apt/apt.conf.d/99-disable-recommends
APT::Install-Recommends "false";
APT::Install-Suggests "false";
APT::AutoRemove::RecommendsImportant "false";
APT::AutoRemove::SuggestsImportant "false";
EOF

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHONDONTWRITEBYTECODE=1
ARG UV_NO_CACHE=1

# set up python
COPY --from=uv /uv /uvx /bin/
COPY .python-version pyproject.toml uv.lock README.md ./
RUN uv python install && \
    uv sync --no-default-groups --no-install-project && \
    chown -R "${USER}:${USER}" "${VIRTUAL_ENV}" && \
    chown -R "${USER}:${USER}" "${APP_HOME}" && \
    uv pip list

# set up project
COPY seq_rec seq_rec
RUN uv sync --no-default-groups

USER ${USER}
HEALTHCHECK NONE
