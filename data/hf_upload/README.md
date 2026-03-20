---
license: other
task_categories:
  - audio-classification
  - audio-to-audio
tags:
  - sound-event-detection
  - target-sound-extraction
  - binaural-audio
  - soundscape
---

# Soundscape Control Data

Datasets for training fine-grained soundscape control models
(Target Sound Extraction + Sound Event Detection).

Paper: "Aurchestra" (MobiSys 2026 #198)

## Hosted Datasets

### DISCO (Domestic Indoor Sound Collection)
- **License**: CC-BY 4.0
- **Attribution**: Zenodo record 4019030
- **Path**: `disco/`

### CIPIC HRTF Database
- **License**: Public Domain (UC Davis)
- **Attribution**: UC Davis CIPIC Interface Lab
- **Path**: `cipic_hrtf/`

## External Datasets (Download Required)

### FSD50K
- **License**: Mixed CC per-clip (CC0/CC-BY/CC-BY-NC)
- **HuggingFace mirror**: [`Fhrozen/FSD50k`](https://huggingface.co/datasets/Fhrozen/FSD50k)
- Setup: `python data/setup_data.py --datasets fsd50k`

### ESC-50
- **License**: CC-BY-NC 3.0 (Karol Piczak)
- **HuggingFace mirror**: [`ashraq/esc50`](https://huggingface.co/datasets/ashraq/esc50)
- Setup: `python data/setup_data.py --datasets esc50`

### musdb18
- **License**: Academic/non-commercial only (custom)
- **Download**: [Zenodo 1117372](https://zenodo.org/records/1117372)
- Requires manual download, then: `python data/setup_data.py --datasets musdb18 --manual_dir <path>`

### TAU Urban Acoustic Scenes 2019
- **License**: Tampere University custom (non-commercial, research only)
- **Download**: [Zenodo 2589280](https://zenodo.org/records/2589280)
- ~35 GB, 21 zip files
- Requires manual download, then: `python data/setup_data.py --datasets tau --manual_dir <path>`

## Quick Start

```bash
# All datasets (auto + manual guide)
python data/setup_data.py --output_dir ./BinauralCuratedDataset

# Specific datasets
python data/setup_data.py --output_dir ./data --datasets fsd50k,esc50,disco,cipic

# With manually downloaded TAU/musdb18
python data/setup_data.py --output_dir ./data --manual_dir ./manual_downloads
```

## Citation

If you use these datasets, please cite the original dataset papers and:

```bibtex
@inproceedings{aurchestra2026,
  title={Aurchestra: Fine-Grained Soundscape Control for Augmented Hearing},
  booktitle={MobiSys},
  year={2026}
}
```
