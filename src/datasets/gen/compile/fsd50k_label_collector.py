import os
import argparse
import json
import urllib.request

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from ontology import Ontology
import numpy as np
import random


# This assumes all samples are leaves in the vocabulary. In general, this is not true.


class FSD50KLabelCollector:
    """
    FSD50K Curator dataset.

    Args:
        root_dir (str): Root directory of the FSD50K dataset.
        onotology (json file): AudioSet ontology file.
    """

    def __init__(self, root_dir, ontology_path):
        self.root_dir = root_dir

        self.ontology = Ontology(ontology_path)

        # Label ratings
        with open(
            os.path.join(root_dir, "FSD50K.metadata", "pp_pnp_ratings_FSD50K.json")
        ) as ratings_file:
            self.pp_pnp_ratings = json.load(ratings_file)

    def is_pp_sample(self, fname):
        # assert fname in self.pp_pnp_ratings.keys(), "fname not in ratings"
        # if id not in self.pp_pnp_ratings[fname].keys():
        #     return False
        # assert id in self.pp_pnp_ratings[fname].keys(), \
        #     "id not in ratings: id=%s fname=%s" % (id, fname)

        label_ratings = self.pp_pnp_ratings[fname]

        for node_id in label_ratings.keys():
            label_rating = label_ratings[node_id]
            counts = {1.0: 0, 0.5: 0, 0: 0, -1: 0}
            for r in label_rating:
                # if r not in counts.keys():
                #     counts[r] = 0
                counts[r] += 1

            if counts[0.0] > 0 or counts[-1] > 0 or counts[1.0] < 2:
                return False

        return True

    def get_sample_split(self):
        dev_samples = pd.read_csv(
            os.path.join(
                self.root_dir, "FSD50K.metadata", "collection", "collection_dev.csv"
            )
        )
        eval_samples = pd.read_csv(
            os.path.join(
                self.root_dir, "FSD50K.metadata", "collection", "collection_eval.csv"
            )
        )

        train = dev_samples
        test = eval_samples

        return train, test

    def _curate_samples(self, samples, exclude=[]):
        # Format data
        samples = samples.dropna().copy()
        samples["fname"] = samples["fname"].apply(lambda x: str(x))
        samples["mids"] = samples["mids"].apply(lambda x: x.split(","))
        samples["labels"] = samples["labels"].apply(lambda x: x.split(","))

        # Filter out samples without multiple true-positive ratings
        samples["pp_sample"] = samples.apply(
            lambda x: self.is_pp_sample(x["fname"]), axis=1
        )
        samples = samples[samples["pp_sample"] == True]

        # Remove samples with source ambiguous sounds
        # samples = samples[samples['mids'].apply(
        #    lambda x: not any([self.ontology.is_source_ambiguous(n) for n in x] ))]

        # Remove samples with multiple labels
        samples = samples[samples["mids"].apply(lambda x: len(x) == 1)]

        # Get sample id from mids
        samples["id"] = samples["mids"].apply(lambda x: x[0])

        # Convert ID to AudioSet label
        samples["label"] = samples["id"].apply(lambda x: self.ontology.get_label(x))

        return samples

    def _write_samples(self, dset, output, exclude=[]):
        if dset == "train" or dset == "val":
            src_dir = "FSD50K.dev_audio"
        elif dset == "test":
            src_dir = "FSD50K.eval_audio"
        samples = self._curate_samples(dset=dset, exclude=exclude)
        samples = samples[["label", "fname", "id"]]
        samples.columns = ["label", "fname", "id"]
        samples["fname"] = samples["fname"].apply(
            lambda x: os.path.join(src_dir, "%s.wav" % x)
        )
        samples.to_csv(output)

    def _plot_stats(self, dset="train", exclude=[], figsize=(20, 5)):
        samples = self._curate_samples(dset=dset, exclude=exclude)
        samples["root_label"] = samples["leaf_id"].apply(
            lambda x: self.ontology[self.ontology.get_ancestor_ids(x)[0]]["name"]
        )
        print("Sample count: %d" % len(samples))
        print("Leaf node count: %d" % len(samples["leaf_label"].unique()))
        plt.figure(figsize=figsize)
        samples["leaf_label"].value_counts().plot(kind="bar")
        plt.grid(True)
        plt.show()
        plt.figure(figsize=figsize)
        samples["root_label"].value_counts().plot(kind="bar")
        plt.grid(True)
        plt.show()

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

        test_samples = self._curate_samples(test)

        train_src_dir = "FSD50K.dev_audio"
        train_samples = train_samples[["label", "fname", "id"]]
        train_samples.columns = ["label", "fname", "id"]
        train_samples["fname"] = train_samples["fname"].apply(
            lambda x: os.path.join(train_src_dir, "%s.wav" % x)
        )

        val_src_dir = "FSD50K.dev_audio"
        val_samples = val_samples[["label", "fname", "id"]]
        val_samples.columns = ["label", "fname", "id"]
        val_samples["fname"] = val_samples["fname"].apply(
            lambda x: os.path.join(val_src_dir, "%s.wav" % x)
        )

        test_src_dir = "FSD50K.eval_audio"
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
            set(train_samples["label"].unique())
            & set(val_samples["label"].unique())
            & set(test_samples["label"].unique())
        )

        # Filter out samples with labels that are not common across
        # train, val and test.
        train_samples = train_samples[train_samples["label"].isin(common_labels)]
        val_samples = val_samples[val_samples["label"].isin(common_labels)]
        test_samples = test_samples[test_samples["label"].isin(common_labels)]

        return train_samples, val_samples, test_samples

    def write_samples(self, output_dir):
        train_samples, val_samples, test_samples = self.curate_samples()
        train_samples.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_samples.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_samples.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    def plot_stats(self, figsize=(20, 5), save_path=None):
        train_samples, val_samples, test_samples = self.curate_samples()
        print(list(train_samples["label"].unique()))
        print("Train sample count: %d" % len(train_samples))
        print("Val sample count: %d" % len(val_samples))
        print("Test sample count: %d" % len(test_samples))
        print("Train leaf node count: %d" % len(train_samples["label"].unique()))

        # print each labels count in train, val, test
        train_counts = train_samples["label"].value_counts()
        val_counts = val_samples["label"].value_counts()
        test_counts = test_samples["label"].value_counts()
        
        print("Train labels count:")
        print(train_counts.to_string())
        print("Val labels count:")
        print(val_counts.to_string())
        print("Test labels count:")
        print(test_counts.to_string())

        plt.figure(figsize=figsize)
        train_counts.plot(kind="bar")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path + "train.png")
        else:
            plt.show()
        print("Val leaf node count: %d" % len(val_samples["label"].unique()))
        plt.figure(figsize=figsize)
        val_counts.plot(kind="bar")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path + "val.png")
        else:
            plt.show()
        print("Test leaf node count: %d" % len(test_samples["label"].unique()))
        plt.figure(figsize=figsize)
        test_counts.plot(kind="bar")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path + "test.png")
        else:
            plt.show()

        # save to csv - create label count summary
        all_labels = set(train_counts.index) | set(val_counts.index) | set(test_counts.index)
        label_counts_list = []
        
        for label in sorted(all_labels):
            train_count = train_counts.get(label, 0)
            val_count = val_counts.get(label, 0)
            test_count = test_counts.get(label, 0)
            
            label_counts_list.append({
                'label': label,
                'train_count': train_count,
                'val_count': val_count,
                'test_count': test_count,
                'total_count': train_count + val_count + test_count
            })
        
        counts_df = pd.DataFrame(label_counts_list)
        # Save CSVs in the dataset directory (not stats subdirectory)
        counts_df.to_csv(os.path.join(self.root_dir, "label_counts.csv"), index=False)
        
        # Also save individual dataset CSVs
        train_samples.to_csv(os.path.join(self.root_dir, "train_counts.csv"), index=False)
        val_samples.to_csv(os.path.join(self.root_dir, "val_counts.csv"), index=False)
        test_samples.to_csv(os.path.join(self.root_dir, "test_counts.csv"), index=False)

        # copy csv to datasets/gen/counts/
        import shutil
        copy_path = "src/datasets/gen/counts/FSD50K"
        if not os.path.exists(copy_path):
            os.makedirs(copy_path, exist_ok=True)
        shutil.copy(os.path.join(args.dataset_dir, "label_counts.csv"), os.path.join(copy_path, "label_counts.csv"))
        shutil.copy(os.path.join(args.dataset_dir, "train_counts.csv"), os.path.join(copy_path, "train_counts.csv"))
        shutil.copy(os.path.join(args.dataset_dir, "val_counts.csv"), os.path.join(copy_path, "val_counts.csv"))
        shutil.copy(os.path.join(args.dataset_dir, "test_counts.csv"), os.path.join(copy_path, "test_counts.csv"))


def main(args):
    random.seed(0)
    np.random.seed(0)

    # assert not os.path.exists(args.output_dir), \
    #     "Ouput dir %s already exists" % args.output_dir
    # os.makedirs(args.output_dir, exist_ok=True)
    fsd50k_curator = FSD50KLabelCollector(args.dataset_dir, args.ontology_path)
    fsd50k_curator.write_samples()

def analyze_dataset(args):
    random.seed(0)
    np.random.seed(0)
    save_path = os.path.join(args.dataset_dir, "stats")
    os.makedirs(save_path, exist_ok=True)
    fsd50k_curator = FSD50KLabelCollector(args.dataset_dir, args.ontology_path)
    fsd50k_curator.plot_stats(save_path=save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology_path", type=str, default="ontology.json")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/BinauralCuratedDataset/FSD50K",
        help="Root directory for the FSD50K dataset",
    )
    parser.add_argument(
        "--analyze_dataset", action="store_true", help="Analyze dataset.",
    )
    args = parser.parse_args()

    if args.analyze_dataset:
        analyze_dataset(args)
    else:
        main(args)


