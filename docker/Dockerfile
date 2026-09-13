# Qorix: RL post-training for LLMs
#
# This image layers code on top of qorix-base (which has vLLM, APEX,
# Transformer Engine pre-built). It builds in minutes, not hours.
#
# Build:
#   docker build -f docker/Dockerfile -t qorix .
#
# Run:
#   docker run --gpus all --ipc=host --network=host --ulimit nofile=65536:65536 -it qorix bash
#   docker run --gpus all --ipc=host --network=host --ulimit nofile=65536:65536 qorix train --config configs/run.yaml
#
# Install optional packages inside the container:
#   uv add math-verify
#   uv add megatron-core megatron-bridge

ARG BASE_IMAGE=ghcr.io/eduardoslonski/qorix-base:latest
FROM ${BASE_IMAGE}

# ── Copy project files ──────────────────────────────────────────────────────
WORKDIR /root/qorix
COPY pyproject.toml uv.lock train.py eval.py README.md ./
COPY src/ src/
COPY configs/ configs/
COPY tools/ tools/

# ── Install qorix ───────────────────────────────────────────────────────
# Pre-sync .venv so `uv run` doesn't re-sync at runtime.
# Enable system-site-packages so CUDA packages (apex, TE)
# installed earlier into system Python remain visible in the venv.
RUN ulimit -n 65536 && uv sync --frozen --no-dev \
    && sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg

ENV PATH="/root/qorix/.venv/bin:$PATH"

# ── Persist env vars for SSH sessions ─────────────────────────────────────
# Docker ENV vars aren't inherited by SSH login shells; write them to .bashrc
# so interactive sessions (e.g. RunPod SSH) get the venv PATH and GPU vars.
RUN echo 'export PATH="/root/qorix/.venv/bin:$PATH"' >> /root/.bashrc

ENTRYPOINT []
CMD ["/bin/bash"]
