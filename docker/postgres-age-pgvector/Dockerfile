# syntax=docker/dockerfile:1

ARG AGE_IMAGE=apache/age:latest
FROM ${AGE_IMAGE}

ARG PGVECTOR_VERSION=v0.8.2

USER root

RUN set -eux; \
    printf '%s\n' \
        'Types: deb' \
        'URIs: http://mirrors.tuna.tsinghua.edu.cn/debian' \
        'Suites: trixie trixie-updates trixie-backports' \
        'Components: main contrib non-free non-free-firmware' \
        'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
        '' \
        'Types: deb' \
        'URIs: http://security.debian.org/debian-security' \
        'Suites: trixie-security' \
        'Components: main contrib non-free non-free-firmware' \
        'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
        > /etc/apt/sources.list.d/debian.sources; \
    PG_MAJOR="$(pg_config --version | sed -E 's/.* ([0-9]+).*/\1/')"; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        "postgresql-server-dev-${PG_MAJOR}"; \
    git clone --branch "${PGVECTOR_VERSION}" --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector; \
    make -C /tmp/pgvector; \
    make -C /tmp/pgvector install; \
    rm -rf /tmp/pgvector; \
    apt-get purge -y --auto-remove build-essential git; \
    rm -rf /var/lib/apt/lists/*

COPY docker/postgres-age-pgvector/initdb/ /docker-entrypoint-initdb.d/

USER postgres
