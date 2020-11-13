#!/bin/bash

docker run \
    -it \
    --mount type=bind,src="$(pwd)"/datasets,dst=/ennclave-experiments/datasets \
    --mount type=bind,src="$(pwd)"/models,dst=/ennclave-experiments/models \
    --mount type=bind,src="$(pwd)"/timing_logs,dst=/ennclave-experiments/timing_logs \
    ennclave-experiments