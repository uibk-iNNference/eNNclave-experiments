#!/bin/bash

(cd $ENNCLAVE_HOME/build && cmake ..)
pip install -e /eNNclave/frontend/python