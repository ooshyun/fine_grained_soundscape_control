import os, sys, glob
import argparse
import random
import json
import pandas as pd
import numpy as np
from ontology import Ontology
from sklearn.model_selection import train_test_split


dictionary = {
    "baby": "Baby cry, infant cry",
    "blender": "Blender",
    "dishwasher": None,
    "electric_shaver_toothbrush": "Toothbrush",
    "fan": "Mechanical fan",
    "frying": "Frying (food)",
    "printer": "Printer",
    "vacuum_cleaner": "Vacuum cleaner",
    "washing_machine": None,
    "water": "Water",
}


class DiscoNoiseLabelCollector:
    def __init__(self, dataset_dir, ontology_path) -> None:
        self.ontology = Ontology(ontology_path)
        self.dataset_dir = dataset_dir

        self.files = {}

        for label in os.listdir(os.path.join(dataset_dir, "train")):
            label_dir = os.path.join(dataset_dir, "train", label)
            for x in glob.glob(os.path.join(label_dir, "*")):
                if label not in self.files:
                    self.files[label] = []

                self.files[label].append(x)

        for label in os.listdir(os.path.join(dataset_dir, "test")):
            label_dir = os.path.join(dataset_dir, "test", label)
            for x in glob.glob(os.path.join(label_dir, "*")):
                if label not in self.files:
                    self.files[label] = []

                self.files[label].append(x)

    def write_samples(self):
        train = []
        test = []
        val = []

        for label in self.files:
            audio_set_label = dictionary[label]

            # Skip labels with no AudioSet equivalent
            if audio_set_label is None:
                continue

            _id = self.ontology.get_id_from_name(audio_set_label)

            train_files, test_files = train_test_split(
                self.files[label], test_size=0.33
            )

            random.shuffle(train_files)
            val_split = int(round(0.1 * len(train_files)))

            val_files = train_files[:val_split]
            train_files = train_files[val_split:]

            train.extend(
                [
                    dict(
                        id=_id,
                        label=audio_set_label,
                        fname=os.path.relpath(fname, self.dataset_dir),
                    )
                    for fname in train_files
                ]
            )
            test.extend(
                [
                    dict(
                        id=_id,
                        label=audio_set_label,
                        fname=os.path.relpath(fname, self.dataset_dir),
                    )
                    for fname in test_files
                ]
            )
            val.extend(
                [
                    dict(
                        id=_id,
                        label=audio_set_label,
                        fname=os.path.relpath(fname, self.dataset_dir),
                    )
                    for fname in val_files
                ]
            )

        train = pd.DataFrame.from_records(train)
        val = pd.DataFrame.from_records(val)
        test = pd.DataFrame.from_records(test)

        train.to_csv(os.path.join(self.dataset_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(self.dataset_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(self.dataset_dir, "test.csv"), index=False)


    def analyze_dataset(self):
        train = {}
        test = {}
        val = {}

        for label in self.files:
            audio_set_label = dictionary[label]
            
            # Skip labels with no AudioSet equivalent
            if audio_set_label is None:
                continue

            _id = self.ontology.get_id_from_name(audio_set_label)

            train_files, test_files = train_test_split(
                self.files[label], test_size=0.33
            )

            random.shuffle(train_files)
            val_split = int(round(0.1 * len(train_files)))

            val_files = train_files[:val_split]
            train_files = train_files[val_split:]

            if label not in train.keys():
                train[label] = []

            for fname in train_files:
                train[label].append(dict(
                id=_id,
                label=audio_set_label,
                fname=os.path.relpath(fname, self.dataset_dir),
                ))                

            if label not in test.keys():
                test[label] = []

            for fname in test_files:
                test[label].append(dict(
                id=_id,
                label=audio_set_label,
                fname=os.path.relpath(fname, self.dataset_dir),
            ))                

            if label not in val.keys():
                val[label] = []

            for fname in val_files:
                val[label].append(dict(
                id=_id,
                label=audio_set_label,
                fname=os.path.relpath(fname, self.dataset_dir),
            ))                


        # count samples
        print("Train labels count:")
        for label, samples in train.items():
            print(f"{label}: {len(samples)}")
        print("Val labels count:")
        for label, samples in val.items():
            print(f"{label}: {len(samples)}")
        print("Test labels count:")
        for label, samples in test.items():
            print(f"{label}: {len(samples)}")

        # save to json
        with open(os.path.join(self.dataset_dir, "train.json"), "w") as f:
            json.dump(train, f, indent=2)
        with open(os.path.join(self.dataset_dir, "test.json"), "w") as f:
            json.dump(test, f, indent=2)
        with open(os.path.join(self.dataset_dir, "val.json"), "w") as f:
            json.dump(val, f, indent=2)


        # save to csv - create label count summary
        label_counts = []
        all_labels = set(train.keys()) | set(val.keys()) | set(test.keys())
        
        for label in sorted(all_labels):
            train_count = len(train.get(label, []))
            val_count = len(val.get(label, []))
            test_count = len(test.get(label, []))
            
            # Get the AudioSet label name for this label
            audio_set_label = dictionary.get(label, label)
            
            label_counts.append({
                'label': label,
                'audio_set_label': audio_set_label,
                'train_count': train_count,
                'val_count': val_count,
                'test_count': test_count,
                'total_count': train_count + val_count + test_count
            })
        
        counts_df = pd.DataFrame(label_counts)
        counts_df.to_csv(os.path.join(self.dataset_dir, "label_counts.csv"), index=False)
        
        # Also save individual dataset CSVs (flatten the lists)
        train_list = []
        for label, samples in train.items():
            train_list.extend(samples)
        train_df = pd.DataFrame(train_list)
        train_df.to_csv(os.path.join(self.dataset_dir, "train_counts.csv"), index=False)
        
        val_list = []
        for label, samples in val.items():
            val_list.extend(samples)
        val_df = pd.DataFrame(val_list)
        val_df.to_csv(os.path.join(self.dataset_dir, "val_counts.csv"), index=False)
        
        test_list = []
        for label, samples in test.items():
            test_list.extend(samples)
        test_df = pd.DataFrame(test_list)
        test_df.to_csv(os.path.join(self.dataset_dir, "test_counts.csv"), index=False)
        
        # copy csv to datasets/gen/counts/
        import shutil
        copy_path = "src/datasets/gen/counts/disco_noises"
        if not os.path.exists(copy_path):
            os.makedirs(copy_path, exist_ok=True)
        shutil.copy(os.path.join(args.dataset_dir, "label_counts.csv"), os.path.join(copy_path, "label_counts.csv"))
        shutil.copy(os.path.join(args.dataset_dir, "train_counts.csv"), os.path.join(copy_path, "train_counts.csv"))
        shutil.copy(os.path.join(args.dataset_dir, "val_counts.csv"), os.path.join(copy_path, "val_counts.csv"))
        shutil.copy(os.path.join(args.dataset_dir, "test_counts.csv"), os.path.join(copy_path, "test_counts.csv"))





def main(args):
    random.seed(0)
    np.random.seed(0)
    label_collector = DiscoNoiseLabelCollector(args.dataset_dir, args.ontology_path)
    label_collector.write_samples()

def analyze_dataset(args):
    random.seed(0)
    np.random.seed(0)
    label_collector = DiscoNoiseLabelCollector(args.dataset_dir, args.ontology_path)
    label_collector.analyze_dataset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology_path", type=str, default="ontology.json")
    parser.add_argument(
        "--dataset_dir", type=str, default="data/BinauralCuratedDataset/disco_noises"
    )
    parser.add_argument(
        "--analyze_dataset", action="store_true", help="Analyze dataset.",
    )
    args = parser.parse_args()

    if args.analyze_dataset:
        analyze_dataset(args)
    else:
        main(args)

