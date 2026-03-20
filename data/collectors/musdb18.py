from __future__ import annotations

import glob
import logging
import os
import random

import ffmpegio
import numpy as np
import pandas as pd
from scipy.io.wavfile import write as wavwrite
from tqdm import tqdm

from data.collectors.ontology import Ontology

logger = logging.getLogger(__name__)


def _write_audio(path: str, data: np.ndarray, sr: int) -> None:
    wavwrite(path, sr, data)


def _convert_videos(
    video_paths: list[str],
    audio_dir: str,
    segment_duration_s: float,
) -> None:
    """Extract vocals and instrumental stems from MUSDB18 STEMS files.

    Each video is split into *segment_duration_s*-second chunks.  Silent
    chunks (max amplitude < 5e-3) are discarded.
    """
    os.makedirs(audio_dir, exist_ok=True)

    instrumental_dir = os.path.join(audio_dir, "instrumental")
    os.makedirs(instrumental_dir, exist_ok=True)

    vocals_dir = os.path.join(audio_dir, "vocals")
    os.makedirs(vocals_dir, exist_ok=True)

    for path in tqdm(video_paths, desc="Converting stems"):
        song_name = os.path.basename(path)
        audio_streams = ffmpegio.probe.audio_streams_basic(path)
        duration_samples = audio_streams[0]["duration"].numerator
        sr = audio_streams[0]["sample_rate"]
        segment_samples = int(round(sr * segment_duration_s))

        # Remaining audio must be at least half a chunk
        num_chunks = (
            1
            + (duration_samples - segment_samples // 2 - 1)
            // segment_samples
        )

        for chunk_id in tqdm(range(num_chunks), leave=False):
            start_time = chunk_id * segment_duration_s
            _, mixture = ffmpegio.audio.read(
                path, ss=start_time, t=segment_duration_s, ac=1
            )
            _, vocals = ffmpegio.audio.read(
                path,
                ss=start_time,
                t=segment_duration_s,
                map=[["0", "4"]],
                ac=1,
            )

            instrumental = mixture - vocals

            if (np.abs(vocals) > 5e-3).any():
                out = os.path.join(vocals_dir, f"{song_name}_v_{chunk_id}.wav")
                _write_audio(out, vocals, sr)

            if (np.abs(instrumental) > 5e-3).any():
                out = os.path.join(
                    instrumental_dir, f"{song_name}_i_{chunk_id}.wav"
                )
                _write_audio(out, instrumental, sr)


class MUSDB18Collector:
    """Collect and curate MUSDB18 into train/val/test CSV splits.

    Expected input layout::

        raw_dir/musdb18/
            train/   (STEMS files)
            test/    (STEMS files)

    The collector first extracts vocals and instrumentals into 15-second
    WAV chunks under ``raw_dir/musdb18/audio/{train,val,test}/``, then
    writes CSV manifests.

    Output::

        output_dir/musdb18/{train,val,test}.csv   (columns: fname, label, id)
    """

    def __init__(self, ontology: Ontology, segment_duration_s: float = 15) -> None:
        self.ontology = ontology
        self.segment_duration_s = segment_duration_s

    def _write_csv(
        self, dataset_dir: str, dataset_type: str, out_dir: str
    ) -> pd.DataFrame:
        samples: list[dict] = []
        preproc_dir = os.path.join(dataset_dir, "audio", dataset_type)

        for sample_path in glob.glob(
            os.path.join(preproc_dir, "vocals", "*.wav")
        ):
            rel_path = os.path.relpath(sample_path, dataset_dir)
            label = "Singing"
            samples.append(
                dict(
                    label=label,
                    fname=rel_path,
                    id=self.ontology.get_id_from_name(label),
                )
            )

        for sample_path in glob.glob(
            os.path.join(preproc_dir, "instrumental", "*.wav")
        ):
            rel_path = os.path.relpath(sample_path, dataset_dir)
            label = "Melody"
            samples.append(
                dict(
                    label=label,
                    fname=rel_path,
                    id=self.ontology.get_id_from_name(label),
                )
            )

        df = pd.DataFrame.from_records(samples)
        df.to_csv(os.path.join(out_dir, f"{dataset_type}.csv"), index=False)
        return df

    def collect(self, raw_dir: str, output_dir: str) -> None:
        dataset_dir = os.path.join(raw_dir, "musdb18")
        out_dir = os.path.join(output_dir, "musdb18")
        os.makedirs(out_dir, exist_ok=True)

        audio_dir = os.path.join(dataset_dir, "audio")

        if not os.path.exists(audio_dir):
            logger.info(
                "MUSDB18: preprocessing stems into WAV chunks "
                "(this may take some time)..."
            )
            os.makedirs(audio_dir, exist_ok=True)

            test_videos = sorted(
                glob.glob(os.path.join(dataset_dir, "test", "*"))
            )
            train_videos = sorted(
                glob.glob(os.path.join(dataset_dir, "train", "*"))
            )

            # 90:10 train:val split
            random.shuffle(train_videos)
            val_split = int(round(0.1 * len(train_videos)))
            val_videos = train_videos[:val_split]
            train_videos = train_videos[val_split:]

            _convert_videos(
                train_videos,
                os.path.join(audio_dir, "train"),
                self.segment_duration_s,
            )
            _convert_videos(
                test_videos,
                os.path.join(audio_dir, "test"),
                self.segment_duration_s,
            )
            _convert_videos(
                val_videos,
                os.path.join(audio_dir, "val"),
                self.segment_duration_s,
            )

        for split in ("train", "val", "test"):
            df = self._write_csv(dataset_dir, split, out_dir)
            logger.info("MUSDB18 %s: %d samples", split, len(df))
