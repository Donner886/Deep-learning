# Base: TensorFlow 2.3.0 CPU image
FROM tensorflow/tensorflow:2.9.1

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System packages commonly needed for scientific Python, dataset tools, and visualization
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    curl \
    ca-certificates \
    wget \
    unzip \
    graphviz \
    # The following X/GL libs support OpenCV headless and some plotting backends
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip toolchain (kept conservative for TF 2.3 era compatibility)
# Use setuptools<65 to avoid wrapt build issues
RUN python -m pip install --upgrade "pip<23" "setuptools<65" wheel

# Workspace directory
WORKDIR /workspace

# Declare a volume mount point; bind a host path to /workspace at runtime
VOLUME ["/workspace"]

# Python packages:
# - Crucial pins for TF 2.3: numpy (<1.19) and h5py (==2.10.0)
# - Jupyter (Lab + classic), ipywidgets, and common data science libs
# - TF ecosystem: tensorflow-addons (0.11.2), tensorflow-datasets (4.0.1), hub
# - Visualization and utilities
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt -v
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix
RUN rm -f /tmp/requirements.txt

# Copy and set up startup script
COPY start-jupyter.sh /usr/local/bin/start-jupyter.sh
RUN chmod +x /usr/local/bin/start-jupyter.sh

# Expose Jupyter port
EXPOSE 8888

# Auto-run Jupyter Notebook on container start using startup script
CMD ["/usr/local/bin/start-jupyter.sh"]