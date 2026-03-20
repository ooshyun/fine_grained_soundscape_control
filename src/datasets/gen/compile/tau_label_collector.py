import os
import glob
import argparse
import json
import urllib.request

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

import tqdm

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TAULabelCollector:
    """
    TAU Curator dataset.

    Args:
        root_dir (str): Root directory of the FSD50K dataset.
    """

    def __init__(self, root_dir):
        # train: (90%) TAU-urban-acoustic-scenes-2019-development
        # val: (10%) TAU-urban-acoustic-scenes-2019-development
        # test: TAU-urban-acoustic-scenes-2019-evaluation
        self.root_dir = root_dir
        self.rng = np.random.RandomState(42)

    def get_sample_split(self):
        dev_samples = glob.glob(
            os.path.join(
                self.root_dir,
                "TAU-urban-acoustic-scenes-2019-development",
                "audio",
                "*.wav",
            )
        )
        eval_samples = glob.glob(
            os.path.join(
                self.root_dir,
                "TAU-urban-acoustic-scenes-2019-evaluation",
                "audio",
                "*.wav",
            )
        )
        return dev_samples, eval_samples

    def _curate_samples(self, samples: list, is_test: bool = False) -> pd.DataFrame:
        """
        Curate samples.
            Extract label and id from filename.
            File example: tram-lisbon-1200-45720-a.wav
            Label: tram-lisbon
            Id: 1200-45720-a

        Args:
            samples (list): List of sample filenames.

        Returns:
            pd.DataFrame: DataFrame with label and id columns.
        """
        samples_processed = pd.DataFrame(
            {"fname": [os.path.basename(x).split(".")[0] for x in samples]}
        )  # remove the extension
        # print(os.path.basename(samples[0]).split('-'))
        # assert False
        if not is_test:
            samples_processed["label"] = samples_processed["fname"].apply(
                lambda x: x.split("-")[0] + "-" + x.split("-")[1]
            )
        else:
            samples_processed["label"] = samples_processed["fname"].apply(
                lambda x: x.split(".")[0]
            )
        # after split idex 2, join the rest of the elements not first one.
        samples_processed["id"] = samples_processed["fname"].apply(
            lambda x: ("-".join(x.split("-")[2:])).split(".")[0]
        )
        return samples_processed[["fname", "label", "id"]]

    def curate_samples(self):
        train, test = self.get_sample_split()

        train_samples_curated = self._curate_samples(train)

        train_samples = pd.DataFrame()
        val_samples = pd.DataFrame()

        train_labels = sorted(list(set(train_samples_curated["label"])))
        for label in train_labels:
            samples = train_samples_curated[train_samples_curated["label"] == label]

            if len(samples) == 1:
                continue

            train, val = train_test_split(samples, test_size=0.1)
            val_samples = pd.concat([val_samples, val])
            train_samples = pd.concat([train_samples, train])

        test_samples = self._curate_samples(test, is_test=True)

        train_src_dir = "TAU-urban-acoustic-scenes-2019-development/audio"
        logger.debug(f"Train source directory: {train_src_dir}")
        logger.debug(f"Train samples:\n {train_samples}")
        train_samples = train_samples[["label", "fname", "id"]]
        train_samples.columns = ["label", "fname", "id"]
        train_samples["fname"] = train_samples["fname"].apply(
            lambda x: os.path.join(train_src_dir, "%s.wav" % x)
        )

        val_src_dir = "TAU-urban-acoustic-scenes-2019-development/audio"
        val_samples = val_samples[["label", "fname", "id"]]
        val_samples.columns = ["label", "fname", "id"]
        val_samples["fname"] = val_samples["fname"].apply(
            lambda x: os.path.join(val_src_dir, "%s.wav" % x)
        )

        test_src_dir = "TAU-urban-acoustic-scenes-2019-evaluation/audio"
        test_samples = test_samples[["label", "fname", "id"]]
        test_samples.columns = ["label", "fname", "id"]
        test_samples["fname"] = test_samples["fname"].apply(
            lambda x: os.path.join(test_src_dir, "%s.wav" % x)
        )

        train_samples = train_samples[
            train_samples["label"].map(train_samples["label"].value_counts()) >= 0
        ]

        # List common labels across train, val and test
        common_labels = list(
            set(train_samples["label"].unique()) & set(val_samples["label"].unique())
        )

        logger.debug(f"Common labels: {common_labels}")

        # Filter out samples with labels that are not common across
        # train, val and test.
        train_samples = train_samples[train_samples["label"].isin(common_labels)]
        val_samples = val_samples[val_samples["label"].isin(common_labels)]
        test_samples = test_samples

        return train_samples, val_samples, test_samples

    def write_samples(self, output_dir):
        train_samples, val_samples, test_samples = self.curate_samples()
        train_samples.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_samples.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_samples.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    def analyze_dataset(self, dataset_dir):
        train_samples, val_samples, test_samples = self.curate_samples()
        # get labes and count
        train_labels_count = train_samples["label"].value_counts()
        val_labels_count = val_samples["label"].value_counts()
        test_labels_count = test_samples["label"].value_counts()
        train_labels = sorted(list(set(train_samples["label"])))
        val_labels = sorted(list(set(val_samples["label"])))
        test_labels = sorted(list(set(test_samples["label"])))
        analyze_dict = {
            "train": {label: count for label, count in train_labels_count.items()},
            "val": {label: count for label, count in val_labels_count.items()},
            "test": {label: count for label, count in test_labels_count.items()},
        }

        # print each label and its count in train, val, test
        print("Train labels count:")
        for label, count in analyze_dict["train"].items():
            print(f"{label}: {count}")
        print("Val labels count:")
        for label, count in analyze_dict["val"].items():
            print(f"{label}: {count}")
        print("Test labels count:")
        for label, count in analyze_dict["test"].items():
            print(f"{label}: {count}")

        # save train, test, val to json
        with open(os.path.join(dataset_dir, "train.json"), "w") as f:
            json.dump(analyze_dict["train"], f, indent=2)
        with open(os.path.join(dataset_dir, "val.json"), "w") as f:
            json.dump(analyze_dict["val"], f, indent=2)
        with open(os.path.join(dataset_dir, "test.json"), "w") as f:
            json.dump(analyze_dict["test"], f, indent=2)

        # save to csv - create label count summary
        all_labels = set(analyze_dict["train"].keys()) | set(analyze_dict["val"].keys()) | set(analyze_dict["test"].keys())
        label_counts = []
        
        for label in sorted(all_labels):
            train_count = analyze_dict["train"].get(label, 0)
            val_count = analyze_dict["val"].get(label, 0)
            test_count = analyze_dict["test"].get(label, 0)
            
            label_counts.append({
                'label': label,
                'train_count': train_count,
                'val_count': val_count,
                'test_count': test_count,
                'total_count': train_count + val_count + test_count
            })
        
        counts_df = pd.DataFrame(label_counts)
        counts_df.to_csv(os.path.join(dataset_dir, "label_counts.csv"), index=False)
        
        # copy csv to datasets/gen/counts/
        import shutil
        copy_path = "src/datasets/gen/counts/tau"
        if not os.path.exists(copy_path):
            os.makedirs(copy_path, exist_ok=True)
        shutil.copy(os.path.join(dataset_dir, "label_counts.csv"), os.path.join(copy_path, "label_counts.csv"))
        
        return analyze_dict

def main(args):
    flag_analyze_dataset = args.analyze_dataset
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    dataset_name = "TAU-acoustic-sounds"
    dataset_path = os.path.join(dataset_dir, dataset_name)

    random.seed(0)
    np.random.seed(0)

    if flag_analyze_dataset:
        tau_curator = TAULabelCollector(dataset_path)
        tau_curator.analyze_dataset(dataset_path)
        return

    assert not os.path.exists(output_dir), "Output dir %s already exists" % output_dir

    dataset_type_list = ["train", "val", "test"]
    is_dataset_csv_exists = True
    for dataset_type in dataset_type_list:
        if not os.path.exists(os.path.join(output_dir, dataset_type)):
            is_dataset_csv_exists = False
            break

    if not is_dataset_csv_exists:
        logger.debug(f"Dataset CSV does not exist, curating samples")
        tau_curator = TAULabelCollector(dataset_path)
        tau_curator.write_samples(dataset_path)

    logger.debug(f"Linking train, val, test to output directory")
    os.makedirs(output_dir, exist_ok=True)

    # link train, val, test to output directory
    for dataset_type in dataset_type_list:
        file_path_csv = os.path.join(dataset_path, f"{dataset_type}.csv")
        if not os.path.exists(file_path_csv):
            assert False, f"File {file_path_csv} does not exist"

        out_dir = os.path.join(output_dir, dataset_type)
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Linking {dataset_type} to {out_dir}")
        samples = pd.read_csv(file_path_csv)
        for index, sample in tqdm.tqdm(samples.iterrows(), total=samples.shape[0]):
            # logger.debug(f"Linking sample: {sample['fname']} to {out_dir}")
            _output_dir = os.path.join(
                out_dir,
                (
                    sample["label"]
                    if type(sample["label"]) == str
                    else str(sample["label"])
                ),
            )
            os.makedirs(_output_dir, exist_ok=True)

            s = os.path.join("..", "..", "..", dataset_name, sample["fname"])
            fname = os.path.join(
                dataset_name.lower() + "_" + os.path.basename(sample["fname"])
            )
            d = os.path.join(_output_dir, fname)

            if os.path.exists(d):
                continue
            os.symlink(s, d)

    # assert counting of samples
    for dataset_type in dataset_type_list:
        downloaded_samples_dir = os.listdir(os.path.join(output_dir, dataset_type))
        num_samples = 0
        for sample_dir in downloaded_samples_dir:
            num_samples += len(
                os.listdir(os.path.join(output_dir, dataset_type, sample_dir))
            )

        logger.debug(f"Number of samples in {dataset_type}: {num_samples}")
        assert num_samples == len(
            pd.read_csv(os.path.join(dataset_path, f"{dataset_type}.csv"))
        ), f"Number of samples in {dataset_type} is not equal to the number of samples in the dataset: {len(pd.read_csv(os.path.join(dataset_path, f'{dataset_type}.csv')))}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/scr/BinauralCuratedDataset/TAU-acoustic-sounds",
        help="Root directory for the TAU dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scr/BinauralCuratedDataset/noise_scaper_fmt",
        help="Output directory for the TAU dataset",
    )

    parser.add_argument(
        "--analyze_dataset", action="store_true", help="Analyze dataset.",
    )

    parser.add_argument(
        "--ontology_path", type=str, default="ontology.json", help="Ontology path",
    )

    args = parser.parse_args()
    main(args)
