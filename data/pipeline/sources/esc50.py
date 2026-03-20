from __future__ import annotations
from pathlib import Path
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from .base import BaseSource

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


class ESC50Source(BaseSource):
    name = "ESC-50"
    key = "esc50"

    def __init__(self, ontology):
        self.ontology = ontology

    def download(self, raw_dir: Path) -> None:
        out = raw_dir / self.name
        if (out / ".done").exists():
            print(f"  [skip] {self.name} already downloaded")
            return
        print("  Loading from HF: ashraq/esc50 ...")
        ds = load_dataset("ashraq/esc50")
        audio_dir = out / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for row in ds["train"]:  # ESC-50 on HF is single split
            fname = row["filename"]
            audio_path = audio_dir / fname
            if not audio_path.exists():
                audio = row["audio"]
                sf.write(str(audio_path), audio["array"], audio["sampling_rate"])
            rows.append({
                "filename": fname,
                "fold": row["fold"],
                "target": row["target"],
                "category": row["category"],
                "esc10": row.get("esc10", False),
            })
        meta_dir = out / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(meta_dir / "esc50.csv", index=False)
        (out / ".done").touch()
        print(f"  ✓ {self.name} downloaded ({len(rows)} samples)")

    def _filter_samples(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        dataset["label"] = dataset["category"].apply(
            lambda x: ESC50_TO_AUDIOSET.get(x)
        )
        dataset = dataset.dropna(subset=["label"]).copy()
        dataset["fname"] = dataset["filename"].apply(
            lambda x: f"audio/{x}"
        )
        dataset["id"] = dataset["label"].apply(
            lambda x: self.ontology.get_id_from_name(x)
        )
        return dataset

    def collect(self, raw_dir: Path, curated_dir: Path) -> None:
        dataset_dir = raw_dir / self.name
        out_dir = curated_dir / self.name

        meta = pd.read_csv(dataset_dir / "meta" / "esc50.csv")

        # Fold-based split: 1-3 train, 4 val, 5 test
        train_meta = meta[meta["fold"] <= 3]
        val_meta = meta[meta["fold"] == 4]
        test_meta = meta[meta["fold"] == 5]

        cols = ["fname", "label", "id"]
        train = self._filter_samples(train_meta)[cols]
        val = self._filter_samples(val_meta)[cols]
        test = self._filter_samples(test_meta)[cols]

        self._write_csvs(out_dir, train, val, test)
        print(
            f"  ESC-50: train={len(train)}  val={len(val)}  test={len(test)}"
        )
