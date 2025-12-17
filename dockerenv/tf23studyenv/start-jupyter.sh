#!/bin/bash
# Startup script for Jupyter Notebook

echo "Starting Jupyter Notebook..."
echo "Workspace: /workspace"
echo "Access at: http://localhost:8888"
echo ""

# Start Jupyter Notebook
exec jupyter notebook \
     --ip=0.0.0.0 \
     --port=8888 \
     --no-browser \
     --allow-root \
     --NotebookApp.token='' \
     --NotebookApp.password='' \
     --notebook-dir=/workspace \
     --NotebookApp.kernel_manager_class=notebook.services.kernels.kernelmanager.MappingKernelManager
