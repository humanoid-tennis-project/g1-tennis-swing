# Build context: the pinned upstream/GVHMR checkout.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime@sha256:0279f7aa29974bf64e61d0ff6e979b41a249b3662a46e30778dbf80b8c99c361
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app/GVHMR
COPY requirements.txt /tmp/gvhmr-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/gvhmr-requirements.txt
COPY hmr4d /app/GVHMR/hmr4d
COPY tools /app/GVHMR/tools
COPY setup.py /app/GVHMR/setup.py
RUN python -m pip install --no-deps -e .
# Static-camera or default SimpleVO inference; DPVO is intentionally not built.
ENTRYPOINT ["python", "tools/demo/demo.py"]
