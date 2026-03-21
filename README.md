# Fine-Grained Soundscape Control for Augmented Hearing

TSE (Target Sound Extraction) and SED (Sound Event Detection) pipelines for binaural augmented hearing with on-the-fly spatial audio synthesis using Head-Related Transfer Functions (HRTF).

Part of **MobiSys 2026 #198 "Aurchestra"**.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Setup dataset

Downloads the public BinauralCuratedDataset tar (~125GB) and builds `noise_scaper_fmt/` for TAU noise backgrounds.

```bash
bash scripts/setup_dataset.sh --output_dir /path/to/output
```

If you already have the tar extracted, or TAU raw data at a separate path:

```bash
# Skip download (tar already extracted)
bash scripts/setup_dataset.sh --output_dir /path/to/output --skip_download

# TAU raw data at separate path
bash scripts/setup_dataset.sh --output_dir /path/to/output \
    --tau_raw_dir /path/to/TAU-2019
```

After setup:
```
/path/to/output/
  BinauralCuratedDataset/
    scaper_fmt/{train,val,test}/{class}/        # foreground audio symlinks
    bg_scaper_fmt/{train,val,test}/{class}/     # background audio symlinks
    noise_scaper_fmt/{train,val,test}/{scene}/  # TAU noise symlinks
    hrtf/CIPIC/{*.sofa, *_hrtf.txt}             # HRTF data
    FSD50K/, ESC-50/, musdb18/, disco_noises/   # raw audio datasets
    TAU-acoustic-sounds/                         # TAU metadata + audio
    start_times.csv                              # silence trimming metadata
```

> **Note on `--data_dir`**: All train/eval scripts expect `--data_dir` to point to
> the **parent** of `BinauralCuratedDataset/`, because the configs reference paths
> like `BinauralCuratedDataset/scaper_fmt/...`. This matches `setup_dataset.sh`'s
> `--output_dir`, so you can pass the same path to both.

#### Datasets & Licenses

| Dataset | License | Source |
|---------|---------|--------|
| [FSD50K](https://zenodo.org/record/4060432) | Mixed CC (CC0/BY/BY-NC) | Zenodo |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | CC-BY-NC 3.0 | GitHub |
| [musdb18](https://sigsep.github.io/datasets/musdb.html) | Academic/non-commercial | Zenodo |
| [DISCO](https://zenodo.org/record/3828141) | CC-BY 4.0 | [HF: ooshyun/fine-grained-soundscape](https://huggingface.co/datasets/ooshyun/fine-grained-soundscape) |
| [TAU-2019](https://zenodo.org/record/2589280) | Tampere Univ. custom (NC) | Zenodo |
| [CIPIC HRTF](https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/) | Public Domain | [HF: ooshyun/fine-grained-soundscape](https://huggingface.co/datasets/ooshyun/fine-grained-soundscape) |

### 3. Train

```bash
# TSE (default: Orange Pi config)
bash scripts/train/run_tse.sh /path/to/output [orange_pi|raspberry_pi|neuralaid]

# SED (default: AST finetune config)
bash scripts/train/run_sed.sh /path/to/output [ast_finetune]
```

### 4. Evaluate (reproduce paper tables)

```bash
# Table 1: TSE model comparison (Orange Pi, Raspberry Pi, NeuralAids)
bash scripts/eval/run_tse.sh /path/to/output

# Table 2: Multi-output TSE (5-out, 20-out)
bash scripts/eval/run_multiout.sh /path/to/output

# Table 3: FiLM ablation (first / all / all-except-first)
bash scripts/eval/run_ablation.sh /path/to/output

# Table 4, Figure 4: SED (Fine-tuned AST)
bash scripts/eval/run_sed.sh /path/to/output
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
│   ├── setup_data.py           # CLI entrypoint for data pipeline
│   ├── class_map.yaml          # Sound class definitions
│   ├── ontology.json           # AudioSet ontology
│   ├── pipeline/               # Modular data pipeline
│   │   ├── download.py         # Stage 1: download datasets
│   │   ├── collect.py          # Stage 2: collect + split CSVs
│   │   ├── prepare.py          # Stage 3: Scaper format + HRTF
│   │   ├── ontology.py         # AudioSet ontology wrapper
│   │   ├── silence.py          # Silence trimming utility
│   │   └── sources/            # Per-dataset logic
│   │       ├── fsd50k.py, esc50.py, disco.py
│   │       ├── cipic.py, musdb18.py, tau.py
│   │       └── base.py         # BaseSource ABC
│   └── hf_upload/              # HuggingFace dataset upload
│       ├── README.md           # Dataset Card
│       └── upload.py           # Upload script
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
│   ├── setup_dataset.sh           # Full dataset setup (download + extract + noise)
│   ├── build_noise_scaper_fmt.py  # Build TAU noise symlinks
│   ├── train/
│   │   ├── run_tse.sh             # Train TSE model
│   │   └── run_sed.sh             # Train SED model
│   └── eval/
│       ├── run_tse.sh             # Table 1: TSE model comparison
│       ├── run_multiout.sh        # Table 2: Multi-output TSE
│       ├── run_ablation.sh        # Table 3: FiLM ablation
│       └── run_sed.sh             # Table 4, Fig 4: SED
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

---

### Appendix: Downloading Raw Datasets Individually

If you need the raw source datasets (e.g. for custom preprocessing), you can download them individually:

```bash
# FSD50K (~59GB, split zip)
# Requires manual merge of split archives
wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip
wget https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip
wget https://zenodo.org/records/4060432/files/FSD50K.metadata.zip

# ESC-50 (~600MB)
wget https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip -O ESC-50.zip

# musdb18 (~5GB, academic use only)
wget https://zenodo.org/records/1117372/files/musdb18.zip

# DISCO (~3GB)
wget https://zenodo.org/api/records/4019030/files/disco_noises.zip/content -O disco_noises.zip

# TAU-2019 (~20GB, 10 parts + meta, non-commercial)
for i in $(seq 1 10); do
  wget "https://zenodo.org/records/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.${i}.zip"
done
wget https://zenodo.org/records/2589280/files/TAU-urban-acoustic-scenes-2019-development.meta.zip

# CIPIC HRTF (~183MB, SOFA files)
# Available from our HuggingFace dataset repo:
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('ooshyun/fine-grained-soundscape', repo_type='dataset', allow_patterns='cipic_hrtf/**', local_dir='.')"
```

Alternatively, use the automated pipeline downloader:

```bash
python data/setup_data.py --output_dir ./data --stage download
python data/setup_data.py --output_dir ./data --datasets fsd50k,esc50 --stage download
python data/setup_data.py --output_dir ./data --dry-run
```
