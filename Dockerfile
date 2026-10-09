# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/uv/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/opt/uv/cache \
    UV_LINK_MODE=copy \
    PATH=/opt/uv/venv/bin:/home/vscode/.local/bin:${PATH}

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        libsndfile1 \
        sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}" \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}" \
    && mkdir -p /workspace/formanta-distill /opt/uv \
    && chown -R "${USER_UID}:${USER_GID}" /workspace/formanta-distill /opt/uv

USER ${USERNAME}
WORKDIR /workspace/formanta-distill

CMD ["sleep", "infinity"]
