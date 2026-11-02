FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /usr/local/bin/

RUN groupadd -g 1000 mobilerun \
    && useradd -m -u 1000 -g 1000 -s /bin/bash mobilerun \
    && install -d -o mobilerun -g mobilerun /mobilerun

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    android-tools-adb \
    && rm -rf /var/lib/apt/lists/*

USER mobilerun

WORKDIR /mobilerun

COPY --chown=mobilerun:mobilerun . .

RUN uv sync --locked --no-dev --no-editable --extra anthropic --no-cache

ENTRYPOINT [".venv/bin/mobilerun"]

CMD ["setup"]
