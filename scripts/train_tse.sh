#!/bin/bash
python -m src.tse.train --config configs/tse/${1:-orange_pi}.yaml "${@:2}"
