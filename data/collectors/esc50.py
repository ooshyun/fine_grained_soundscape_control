from __future__ import annotations

import logging
import os

import pandas as pd

from data.collectors.ontology import Ontology

logger = logging.getLogger(__name__)

# Maps ESC-50 category names to AudioSet label names.
# ``None`` means the category has no AudioSet equivalent and is skipped.
ESC50_TO_AUDIOSET: dict[str, str | None] = {
    "dog": "Bark",
    "rooster": "Crowing, cock-a-doodle-doo",
    "pig": "Pig",
    "cow": "Cattle, bovinae",
    "frog": "Frog",
    "cat": "Meow",
    "hen": "Chicken, rooster",
    "insects": "Insect",
    "sheep": "Sheep",
    "crow": "Crow",
    "rain": "Rain",
    "sea_waves": "Waves, surf",
    "crackling_fire": "Crackle",
    "crickets": "Cricket",
    "chirping_birds": "Chirp, tweet",
    "water_drops": "Drip",
    "wind": "Wind",
    "pouring_water": "Pour",
    "toilet_flush": "Toilet flush",
    "thunderstorm": "Thunderstorm",
    "crying_baby": "Baby cry, infant cry",
    "sneezing": "Sneeze",
    "clapping": "Clapping",
    "breathing": "Breathing",
    "coughing": "Cough",
    "footsteps": "Walk, footsteps",
    "laughing": "Laughter",
    "brushing_teeth": "Toothbrush",
    "snoring": "Snoring",
    "drinking_sipping": None,
    "door_wood_knock": "Knock",
    "mouse_click": None,
    "keyboard_typing": "Computer keyboard",
    "door_wood_creaks": "Creak",
    "can_opening": None,
    "washing_machine": None,
    "vacuum_cleaner": "Vacuum cleaner",
    "clock_alarm": "Alarm clock",
    "clock_tick": "Tick-tock",
    "glass_breaking": "Shatter",
    "helicopter": "Helicopter",
    "chainsaw": "Chainsaw",
    "siren": "Siren",
    "car_horn": "Vehicle horn, car horn, honking",
    "engine": "Engine",
    "train": "Train",
    "church_bells": "Church bell",
    "airplane": "Fixed-wing aircraft, airplane",
    "fireworks": "Fireworks",
    "hand_saw": "Sawing",
}


class ESC50Collector:
    """Collect and curate ESC-50 labels into train/val/test CSV splits.

    Expected input layout::

        raw_dir/ESC-50/
            meta/esc50.csv
            audio/*.wav

    Output::

        output_dir/ESC-50/{train,val,test}.csv   (columns: fname, label, id)
    """

    def __init__(self, ontology: Ontology) -> None:
        self.ontology = ontology

    def _filter_samples(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        dataset["label"] = dataset["category"].apply(
            lambda x: ESC50_TO_AUDIOSET[x]
        )
        dataset = dataset.dropna(subset=["label"]).copy()
        dataset["fname"] = dataset["filename"].apply(
            lambda x: os.path.join("audio", x)
        )
        dataset["id"] = dataset["label"].apply(
            lambda x: self.ontology.get_id_from_name(x)
        )
        return dataset

    def collect(self, raw_dir: str, output_dir: str) -> None:
        # raw_dir should point directly to the ESC-50-master/ directory
        # containing meta/ and audio/
        dataset_dir = raw_dir
        out_dir = os.path.join(output_dir, "ESC-50")
        os.makedirs(out_dir, exist_ok=True)

        meta = pd.read_csv(os.path.join(dataset_dir, "meta", "esc50.csv"))

        # Fold-based split: 1-3 train, 4 val, 5 test
        train_meta = meta[meta["fold"] <= 3]
        val_meta = meta[meta["fold"] == 4]
        test_meta = meta[meta["fold"] == 5]

        cols = ["fname", "label", "id"]
        for split_name, split_df in [
            ("train", train_meta),
            ("val", val_meta),
            ("test", test_meta),
        ]:
            filtered = self._filter_samples(split_df)
            filtered[cols].to_csv(
                os.path.join(out_dir, f"{split_name}.csv"), index=False
            )

        logger.info(
            "ESC-50: train=%d  val=%d  test=%d",
            len(self._filter_samples(train_meta)),
            len(self._filter_samples(val_meta)),
            len(self._filter_samples(test_meta)),
        )
