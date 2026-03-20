import os, sys
import argparse

import pandas as pd
import numpy as np
from ontology import Ontology


dictionary = {
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


def write_csv(dataset, csv_name):
    dataset = dataset[dataset.apply(lambda x: dictionary[x["category"]] is not None)]


class ESC50LabelCollector:
    def __init__(self, dataset_dir, ontology_path) -> None:
        self.ontology = Ontology(ontology_path)

        # Load metadata
        meta = pd.read_csv(os.path.join(dataset_dir, "meta/esc50.csv"))

        # # Create a audio path column
        # meta['audio_path'] = meta['filename'].apply(
        #     lambda x: os.path.join('..', '..', '..', 'ESC-50-master', 'audio', x))

        # Use first 3 folds for training, 4th for validation, 5th for testing
        self.train_meta = meta[meta["fold"] <= 3]
        self.val_meta = meta[meta["fold"] == 4]
        self.test_meta = meta[meta["fold"] == 5]

        self.dataset_dir = dataset_dir

    def filter_samples(self, dataset: pd.DataFrame):
        dataset["label"] = dataset["category"].apply(lambda x: dictionary[x])
        dataset = dataset.dropna().copy()

        dataset["fname"] = dataset["filename"].apply(lambda x: os.path.join("audio", x))
        dataset["id"] = dataset["label"].apply(
            lambda x: self.ontology.get_id_from_name(x)
        )

        return dataset


    def count_samples(self, dataset: pd.DataFrame):
        dataset["original_label"] = dataset["category"]
        dataset["label"] = dataset["original_label"].apply(lambda x: dictionary[x])
        dataset = dataset.dropna().copy()

        dataset["fname"] = dataset["filename"].apply(lambda x: os.path.join("audio", x))
        dataset["id"] = dataset["label"].apply(
            lambda x: self.ontology.get_id_from_name(x)
        )

        return dataset

    def write_samples(self):
        columns = ["fname", "label", "id"]

        train = self.filter_samples(self.train_meta)
        train = train[columns]

        val = self.filter_samples(self.val_meta)
        val = val[columns]

        test = self.filter_samples(self.test_meta)
        test = test[columns]

        train.to_csv(os.path.join(self.dataset_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(self.dataset_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(self.dataset_dir, "test.csv"), index=False)


    def analyze_dataset(self):
        columns = ["original_label", "fname", "label", "id"]

        train = self.count_samples(self.train_meta)
        train = train[columns]

        val = self.count_samples(self.val_meta)
        val = val[columns]

        test = self.count_samples(self.test_meta)
        test = test[columns]

        # print each label and its count in train, val, test
        print("Train labels count:")
        train_counts = train["label"].value_counts()
        for label, count in train_counts.items():
            print(f"{label}: {count}")
        print("Val labels count:")
        val_counts = val["label"].value_counts()
        for label, count in val_counts.items():
            print(f"{label}: {count}")
        print("Test labels count:")
        test_counts = test["label"].value_counts()
        for label, count in test_counts.items():
            print(f"{label}: {count}")

        # save to json
        train.to_json(os.path.join(self.dataset_dir, "train.json"), orient="records")
        val.to_json(os.path.join(self.dataset_dir, "val.json"), orient="records")
        test.to_json(os.path.join(self.dataset_dir, "test.json"), orient="records")

        # save to csv - create label count summary
        all_labels = set(train_counts.index) | set(val_counts.index) | set(test_counts.index)
        label_counts = []
        
        for label in sorted(all_labels):
            train_count = train_counts.get(label, 0)
            val_count = val_counts.get(label, 0)
            test_count = test_counts.get(label, 0)
            
            # Get the original label name from dictionary
            original_label = None
            for orig, mapped in dictionary.items():
                if mapped == label:
                    original_label = orig
                    break
            
            label_counts.append({
                'label': label,
                'original_label': original_label if original_label else '',
                'train_count': train_count,
                'val_count': val_count,
                'test_count': test_count,
                'total_count': train_count + val_count + test_count
            })
        
        counts_df = pd.DataFrame(label_counts)
        counts_df.to_csv(os.path.join(self.dataset_dir, "label_counts.csv"), index=False)
        
        # Also save individual dataset CSVs
        train.to_csv(os.path.join(self.dataset_dir, "train_counts.csv"), index=False)
        val.to_csv(os.path.join(self.dataset_dir, "val_counts.csv"), index=False)
        test.to_csv(os.path.join(self.dataset_dir, "test_counts.csv"), index=False)

        # copy csv to datasets/gen/counts/
        import shutil
        copy_path = "src/datasets/gen/counts/ESC-50"
        if not os.path.exists(copy_path):
            os.makedirs(copy_path, exist_ok=True)
        shutil.copy(os.path.join(args.dataset_dir, "label_counts.csv"), os.path.join(copy_path, "label_counts.csv"))
        shutil.copy(os.path.join(args.dataset_dir, "train_counts.csv"), os.path.join(copy_path, "train_counts.csv"))
        shutil.copy(os.path.join(args.dataset_dir, "val_counts.csv"), os.path.join(copy_path, "val_counts.csv"))
        shutil.copy(os.path.join(args.dataset_dir, "test_counts.csv"), os.path.join(copy_path, "test_counts.csv"))



def main(args):
    label_collector = ESC50LabelCollector(args.dataset_dir, args.ontology_path)
    label_collector.write_samples()

def analyze_dataset(args):
    label_collector = ESC50LabelCollector(args.dataset_dir, args.ontology_path)
    label_collector.analyze_dataset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology_path", type=str, default="ontology.json")
    parser.add_argument(
        "--dataset_dir", type=str, default="data/BinauralCuratedDataset/ESC-50"
    )
    parser.add_argument(
        "--analyze_dataset", action="store_true", help="Analyze dataset.",
    )
    args = parser.parse_args()

    if args.analyze_dataset:
        analyze_dataset(args)
    else:
        main(args)

