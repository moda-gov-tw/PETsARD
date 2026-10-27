#!/bin/bash

exec python -m jupyter lab \
       --ip=0.0.0.0 \
       --port=8888 \
       --no-browser
