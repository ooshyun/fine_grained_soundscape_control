FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jupyterlab

# Copy source code
COPY . .

# Ensure submodule is present (needed for Waveformer baseline)
RUN if [ ! -f third_party/SemanticHearing/src/training/dcc_tf.py ]; then \
        git submodule update --init third_party/SemanticHearing || \
        git clone https://github.com/vb000/SemanticHearing.git third_party/SemanticHearing; \
    fi

# Default data mount point
VOLUME ["/data"]

# Default output mount point
VOLUME ["/output"]

EXPOSE 8888
