#!/bin/bash
python -m src.sed.train --config configs/sed/${1:-ast_finetune}.yaml "${@:2}"
