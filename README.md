# Fine-Grained Soundscape Control for Augmented Hearing

TSE (Target Sound Extraction) and SED (Sound Event Detection) pipelines for binaural augmented hearing with on-the-fly spatial audio synthesis using Head-Related Transfer Functions (HRTF).

Part of **MobiSys 2026 #198 "Aurchestra"**.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets (~110GB)
python data/download.py --output_dir ./raw_datasets

# 3. Prepare binaural dataset
python data/prepare.py --raw_dir ./raw_datasets --output_dir ./BinauralCuratedDataset

# 4. Train TSE
python -m src.tse.train --config configs/tse/orange_pi.yaml

# 5. Train SED
python -m src.sed.train --config configs/sed/ast_finetune.yaml

# 6. Evaluate
python -m src.tse.eval --pretrained ooshyun/semantic_listening --model orange_pi
python -m src.sed.eval --pretrained ooshyun/sound_event_detection --model finetuned_ast
```

## Pretrained Models

| Task | HuggingFace Repository | Models |
|------|------------------------|--------|
| TSE | [ooshyun/semantic_listening](https://huggingface.co/ooshyun/semantic_listening) | 11 models |
| SED | [ooshyun/sound_event_detection](https://huggingface.co/ooshyun/sound_event_detection) | Fine-tuned AST |

See [docs/pretrained_models.md](docs/pretrained_models.md) for full model details, download instructions, and STFT configuration.

### TSE Models

| Name | Architecture | D | H | B | Config |
|------|--------------|---|---|---|--------|
| Orange Pi | TFGridNet | 32 | 64 | 6 | `configs/tse/orange_pi.yaml` |
| Raspberry Pi | TFGridNet | 16 | 64 | 3 | `configs/tse/raspberry_pi.yaml` |
| NeuralAids | TFMLPNet | 32 | 32 | 6 | `configs/tse/neuralaid.yaml` |

### SED Models

| Model | Source | Config |
|-------|--------|--------|
| AST (pretrained) | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | -- |
| Fine-tuned AST | [ooshyun/sound_event_detection](https://huggingface.co/ooshyun/sound_event_detection) | `configs/sed/ast_finetune.yaml` |

## Datasets

The training pipeline uses six public datasets synthesized into binaural mixtures on-the-fly:

| Dataset | Description |
|---------|-------------|
| [FSD50K](https://zenodo.org/record/4060432) | Freesound Dataset -- 50k clips of diverse sound events |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | Environmental Sound Classification -- 2k clips, 50 classes |
| [musdb18](https://sigsep.github.io/datasets/musdb.html) | Music source separation dataset -- 150 tracks |
| [DISCO](https://zenodo.org/record/3828141) | Diverse Indoor Sound Corpus -- environmental noise recordings |
| [TAU-2019](https://zenodo.org/record/2589280) | TAU Urban Acoustic Scenes 2019 -- urban noise backgrounds |
| [CIPIC HRTF](https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/) | Head-Related Transfer Function database -- 45 subjects, 1250 directions |

## Project Structure

```
fine_grained_soundscape_control_for_augmented_hearing/
├── configs/
│   ├── tse/                    # TSE training configs
│   │   ├── orange_pi.yaml
│   │   ├── raspberry_pi.yaml
│   │   └── neuralaid.yaml
│   └── sed/                    # SED training configs
│       └── ast_finetune.yaml
├── data/
│   ├── download.py             # Dataset downloader
│   ├── prepare.py              # Binaural dataset preparation
│   ├── class_map.yaml          # Sound class definitions
│   ├── ontology.json           # AudioSet ontology
│   └── collectors/             # Per-dataset download/processing
│       ├── fsd50k.py
│       ├── esc50.py
│       ├── musdb18.py
│       ├── disco_noise.py
│       ├── tau.py
│       └── ontology.py
├── src/
│   ├── datasets/
│   │   ├── soundscape_dataset.py   # Binaural mixture dataset
│   │   ├── hrtf.py                 # HRTF spatialization
│   │   └── augmentations.py        # Audio augmentations
│   ├── trainer/
│   │   ├── base.py                 # Base trainer interface
│   │   ├── lightning.py            # PyTorch Lightning backend
│   │   └── fabric.py              # Lightning Fabric backend
│   ├── metrics/
│   │   ├── tse.py                  # SI-SNRi, SDRi
│   │   └── sed.py                  # mAP, F1
│   ├── tse/
│   │   ├── model.py                # TFGridNet, TFMLPNet
│   │   ├── loss.py                 # TSE losses
│   │   ├── train.py                # TSE training entry
│   │   └── eval.py                 # TSE evaluation entry
│   └── sed/
│       ├── model.py                # AST wrapper
│       ├── loss.py                 # SED losses
│       ├── train.py                # SED training entry
│       └── eval.py                 # SED evaluation entry
├── scripts/
│   ├── train_tse.sh
│   ├── eval_tse.sh
│   ├── train_sed.sh
│   └── eval_sed.sh
├── requirements.txt
└── README.md
```

## Trainer Backends

The training pipeline supports two backends, configurable via the YAML config:

```yaml
training:
  backend: "lightning"   # or "fabric"
```

- **Lightning** (`src/trainer/lightning.py`): Full PyTorch Lightning Trainer with built-in logging, checkpointing, and multi-GPU support. Recommended for standard training.
- **Fabric** (`src/trainer/fabric.py`): Lightweight Lightning Fabric backend with manual training loop control. Useful for custom training logic or debugging.

Both backends share the same base interface (`src/trainer/base.py`) and are interchangeable without modifying model or dataset code.

## Citation

```bibtex
@inproceedings{aurchestra2026,
  title     = {Aurchestra: Fine-Grained Soundscape Control for Augmented Hearing},
  booktitle = {Proceedings of the 24th ACM International Conference on
               Mobile Systems, Applications, and Services (MobiSys '26)},
  year      = {2026},
}
```

## License

MIT
