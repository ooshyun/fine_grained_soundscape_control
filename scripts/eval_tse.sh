#!/bin/bash
python -m src.tse.eval --config configs/tse/${1:-orange_pi}.yaml "${@:2}"
