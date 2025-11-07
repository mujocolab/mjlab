#!/usr/bin/env sh
#
# Injects useful arguments for running mjlab in docker.
# See docs/installation_guide.md for usage.
#
# Patterned after the uv-in-docker example:
# https://github.com/astral-sh/uv-docker-example/blob/5748835918ec293d547bbe0e42df34e140aca1eb/run.sh
#
# Key arguments:
#   --rm                Remove the container after exiting
#   --runtime=nvidia    Use NVIDIA Container runtime to give GPU access
#   --gpus all          Expose all GPUs by default
#   -v .:/app           Mount current directory to /app (code changes don't require rebuild)
#   -v /app/.venv       Mount venv separately (keeps developer's environment out of container)
#   -p 8080:8080        Publish port 8080 for viewing mjlab web interface on the host
#   -it (conditional)   Launch in interactive mode if running in a terminal
#                       (Note: if running training, there's a blocking wandb prompt before training begins)
#   docker build        Build and launch the image (tag matches the Makefile)
#   "$@"                Forward all arguments to the docker image


if [ -t 1 ]; then
    INTERACTIVE="-it"
else
    INTERACTIVE=""
fi

docker run \
    --rm \
    --runtime=nvidia \
    --gpus all \
    --volume .:/app \
    --volume /app/.venv \
    --publish 8080:8080 \
    $INTERACTIVE \
    $(docker build -qt mjlab .) \
    "$@"
