#!/bin/bash
# Startup docker containers for local development

echo "Starting docker container..."

## try to check if the container is already running
CONTAINER_NAME="tf23studyenv_local"
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} is already running."
    exit 0
fi  
## if not running, start a new container
docker run --rm \
    -p 8888:8888 \
    -v /Users/madongdong/Deep-learning:/workspace \
    --name tf23studyenv_local tf29:v0.0.2 