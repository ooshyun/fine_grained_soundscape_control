# Pretrained Models

All pretrained models for **MobiSys 2026 #198 "Aurchestra"**.

## TSE (Target Sound Extraction)

Hosted on HuggingFace: [ooshyun/semantic_listening](https://huggingface.co/ooshyun/semantic_listening)

### STFT Configuration

All TSE models share the same STFT windowing:

| Parameter | Paper Symbol | Value | Samples @16kHz |
|-----------|-------------|-------|----------------|
| stft_chunk_size | L_C | 6 ms | 96 |
| stft_pad_size (lookahead) | L_F | 4 ms | 64 |
| stft_back_pad (lookback) | L_B | 6 ms | 96 |
| nfft | -- | 256 | -- |
| Algorithmic latency | -- | 10 ms | -- |

### Table 1: Model Comparison (1ch, 1out, single source, 16kHz)

| Paper Name | Architecture | D | H | B | Run Dir |
|------------|-------------|---|---|---|---------|
| Orange Pi | TFGridNet | 32 | 64 | 6 | `tfgridnet_large_..._1ch_1spk_1out_..._film_all_except_first_onflight` |
| Raspberry Pi | TFGridNet | 16 | 64 | 3 | `tfgridnet_small_..._1ch_1spk_1out_..._film_all_except_first_onflight` |
| NeuralAids | TFMLPNet | 32 | 32 | 6 | `tfmlpnet_..._1ch_1spk_1out_..._film_all_except_first_onflight` |
| Waveformer | Waveformer | -- | -- | -- | `waveformer_..._1ch_1spk_1out_..._256chunk_film_all_onflight` |

### Table 2: Multi-output (Orange Pi, 5ch)

| Config | Run Dir |
|--------|---------|
| 5-out | `tfgridnet_large_..._5ch_5spk_5out_..._film_all_except_first_onflight` |
| 20-out | `tfgridnet_large_..._20ch_5spk_20out_..._film_all_except_first_onflight` |

### Table 3: FiLM Ablation (5ch, 5spk)

| Model | FiLM Preset | Run Dir |
|-------|-------------|---------|
| Orange Pi | first | `tfgridnet_large_..._5ch_..._film_first_onflight` |
| Orange Pi | all | `tfgridnet_large_..._5ch_..._film_all_onflight` |
| Orange Pi | all-ex-1st | `tfgridnet_large_..._5ch_..._film_all_except_first_onflight` |
| NeuralAids | first | `tfmlpnet_..._5ch_..._film_first_onflight` |
| NeuralAids | all | `tfmlpnet_..._5ch_..._film_all_layers_6_onflight` |
| NeuralAids | all-ex-1st | `tfmlpnet_..._5ch_..._film_all_except_first_onflight` |

### Download TSE Models

```python
from huggingface_hub import hf_hub_download

# Example: Orange Pi (Table 1)
model_name = "tfgridnet_large_snr_ctl_v2_1ch_1spk_1out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight"

checkpoint = hf_hub_download(
    repo_id="ooshyun/semantic_listening",
    filename=f"{model_name}/checkpoints/best.pt",
)
config = hf_hub_download(
    repo_id="ooshyun/semantic_listening",
    filename=f"{model_name}/config.json",
)
```

To list all available models:

```python
from huggingface_hub import list_repo_tree

for item in list_repo_tree("ooshyun/semantic_listening"):
    if item.path.endswith("/config.json"):
        print(item.path.split("/")[0])
```

## SED (Sound Event Detection)

Hosted on HuggingFace: [ooshyun/sound_event_detection](https://huggingface.co/ooshyun/sound_event_detection)

### Models (Table 4, Figure 4)

| Model | Source | Classes | Checkpoint |
|-------|--------|---------|------------|
| YAMNet | [google/yamnet](https://huggingface.co/google/yamnet) | 521 AudioSet | HuggingFace (no local ckpt) |
| AST | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | 527 AudioSet | HuggingFace (no local ckpt) |
| Fine-tuned AST | Based on MIT/AST, fine-tuned on binaural mixtures | 20 target classes | `sed_ast_snr_ctl_v2_16k/checkpoints/best.pt` (~1GB) |

### Download Fine-tuned AST

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="ooshyun/sound_event_detection",
    filename="sed_ast_snr_ctl_v2_16k/checkpoints/best.pt",
)
config = hf_hub_download(
    repo_id="ooshyun/sound_event_detection",
    filename="sed_ast_snr_ctl_v2_16k/config.json",
)
```
