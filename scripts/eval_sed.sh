#!/bin/bash
python -m src.sed.eval --config configs/sed/${1:-ast_finetune}.yaml "${@:2}"
