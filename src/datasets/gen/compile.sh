#!/bin/bash
root=/mnt/sda1/tmp
python compile/consolidate_datasets.py --datasets_dir $root/BinauralCuratedDataset \
--class_definitions Classes.yaml --ontology ontology.json \
--fg_output_dir $root/BinauralCuratedDataset/scaper_fmt \
--bg_output_dir $root/BinauralCuratedDataset/bg_scaper_fmt \
--dry_run