# docker/build_env.py
"""
Helper to generate a Dockerfile for the project environment.
Writes a simple Dockerfile to the target directory.
"""
import textwrap
import os

DOCKERFILE_TEMPLATE = textwrap.dedent("""\
FROM python:3.10-slim

# Install system deps (adjust as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential git curl && \\
    rm -rf /var/lib/apt/lists/*

# Create working dir
WORKDIR /app
COPY . /app

# Install pip packages
RUN pip install --no-cache-dir numpy scipy scikit-learn matplotlib torch torchvision

# (Optional) install faiss/umap if needed:
# RUN pip install faiss-cpu umap-learn

CMD ["python", "run_pipeline.py"]
""")

def write_dockerfile(target_dir='.', filename='Dockerfile'):
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, filename)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(DOCKERFILE_TEMPLATE)
    print(f"Wrote Dockerfile to {path}")
