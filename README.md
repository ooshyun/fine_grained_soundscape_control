# Fine-Grained Soundscape Control for Augmented Hearing

TSE (Target Sound Extraction) and SED (Sound Event Detection) pipelines for binaural augmented hearing with on-the-fly spatial audio synthesis using Head-Related Transfer Functions (HRTF).

Part of **MobiSys 2026 #198 "Aurchestra"**.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Download datasets (~88GB)

Downloads 6 public audio datasets. CIPIC HRTF and DISCO are fetched from our [HuggingFace dataset repo](https://huggingface.co/datasets/ooshyun/fine-grained-soundscape).

```bash
python data/download.py --output_dir ./raw_datasets
```

| Dataset | Source | Size |
|---------|--------|------|
| FSD50K | Zenodo (split zip) | ~59GB |
| ESC-50 | GitHub | ~1GB |
| musdb18 | Zenodo | ~5GB |
| DISCO | HuggingFace | ~3GB |
| TAU-2019 | Zenodo (10 parts) | ~19GB |
| CIPIC HRTF | HuggingFace (45 SOFA) | ~183MB |

### 3. Prepare binaural dataset

Runs per-dataset label collectors and consolidates into Scaper format with symlinks.

```bash
python data/prepare.py --raw_dir ./raw_datasets --output_dir ./BinauralCuratedDataset
```

Note: musdb18 requires `ffmpeg` for STEMS extraction. If unavailable, music/singing classes from musdb18 will be skipped (FSD50K samples still included).

### 4. Evaluate with pretrained models

```bash
# TSE: Orange Pi 5ch, FiLM=All blocks (Table 3)
python -m src.tse.eval \
  --pretrained ooshyun/semantic_listening \
  --model orange_pi_5ch_film_all \
  --data_dir ./BinauralCuratedDataset

# SED: Fine-tuned AST
python -m src.sed.eval \
  --pretrained ooshyun/sound_event_detection \
  --model finetuned_ast \
  --data_dir ./BinauralCuratedDataset
```

### 5. Train from scratch

```bash
python -m src.tse.train --config configs/tse/orange_pi.yaml --data_dir ./BinauralCuratedDataset
python -m src.sed.train --config configs/sed/ast_finetune.yaml --data_dir ./BinauralCuratedDataset
```

## Reproducing Paper Results

The evaluation pipeline reproduces the metrics from **Table 3** of the paper (Orange Pi, FiLM=All blocks, 5 output channels, 1-5 targets in mixture):

| Metric | This Repo | Paper |
|--------|-----------|-------|
| **SNRi (dB)** | **12.31 ± 4.08** | **12.26 ± 4.38** |
| **SI-SNRi (dB)** | **10.18 ± 5.43** | **10.16 ± 5.72** |

Evaluated on 2000 on-the-fly synthesized test samples with 1-5 target sources, 1-3 interfering sources, and urban noise backgrounds.

## Pretrained Models

| Task | HuggingFace Repository | Models |
|------|------------------------|--------|
| TSE | [ooshyun/semantic_listening](https://huggingface.co/ooshyun/semantic_listening) | 11 models |
| SED | [ooshyun/sound_event_detection](https://huggingface.co/ooshyun/sound_event_detection) | Fine-tuned AST |

See [docs/pretrained_models.md](docs/pretrained_models.md) for full model details, download instructions, and STFT configuration.

### TSE Models

| Name | Architecture | D | H | B | Outputs | FiLM |
|------|--------------|---|---|---|---------|------|
| Orange Pi | TFGridNet | 32 | 64 | 6 | 5 | All |
| Raspberry Pi | TFGridNet | 16 | 64 | 3 | 5 | All |
| NeuralAids | TFMLPNet | 32 | 32 | 6 | 5 | All |

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
│   │   ├── MisophoniaDataset.py    # On-the-fly binaural synthesis (original)
│   │   ├── soundscape_dataset.py   # Simplified dataset interface
│   │   ├── multi_ch_simulator.py   # HRTF spatialization (CIPIC, etc.)
│   │   ├── motion_simulator.py     # Sound source motion
│   │   ├── augmentations/          # Audio augmentations (speed, pitch, etc.)
│   │   └── gen/                    # Dataset generation utilities
│   ├── trainer/
│   │   ├── base.py                 # Base trainer interface
│   │   ├── lightning.py            # PyTorch Lightning backend
│   │   └── fabric.py              # Lightning Fabric backend
│   ├── metrics/
│   │   ├── tse.py                  # SI-SNRi, SNRi, per-channel metrics
│   │   └── sed.py                  # mAP, F1, AUC-ROC, d-prime
│   ├── tse/
│   │   ├── model.py                # Pretrained model loading
│   │   ├── net.py                  # TFGridNet STFT wrapper (original)
│   │   ├── multiflim_guided_tfnet.py  # FiLM-conditioned separator
│   │   ├── gridnet_block.py        # Time-frequency processing block
│   │   ├── loss.py                 # Multi-resolution STFT + L1 loss
│   │   ├── train.py                # TSE training entry
│   │   └── eval.py                 # TSE evaluation entry
│   └── sed/
│       ├── model.py                # Pretrained AST loading
│       ├── ast_hf.py               # HuggingFace AST wrapper (original)
│       ├── loss.py                 # BCE + Focal loss
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
