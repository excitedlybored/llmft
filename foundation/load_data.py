
from datasets import load_dataset, ClassLabel
import os
import numpy as np

#function almost exact copy from paper's github repo
def load_glue_datasets(task_name, use_auth_token,cache_dir=None):
    # Get the datasets: specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if task_name is not None:
        if task_name == "mnli":
            # convert to binary format (remove neutral class)
            raw_datasets = load_dataset(
                "glue", task_name, cache_dir=cache_dir)

            raw_datasets = raw_datasets.filter(
                lambda example: example["label"] != 1)

            # change labels of contradiction examples from 2 to 1
            def change_label(example):
                example["label"] = 1 if example["label"] == 2 else example["label"]
                return example
            raw_datasets = raw_datasets.map(change_label)

            # change features to reflect the new labels
            features = raw_datasets["train"].features.copy()
            features["label"] = ClassLabel(
                num_classes=2, names=['entailment', 'contradiction'], id=None)
            raw_datasets = raw_datasets.cast(
                features)  # overwrite old features
        
        elif task_name == "mnli-original":
            # convert to binary format (merge neutral and contradiction class)
            raw_datasets = load_dataset(
                path="glue", name="mnli", cache_dir=cache_dir)

            # change labels of contradiction examples from 2 to 1
            def change_label(example):
                example["label"] = 1 if example["label"] == 2 else example["label"]
                return example
            raw_datasets = raw_datasets.map(change_label)

            # change features to reflect the new labels
            features = raw_datasets["train"].features.copy()
            features["label"] = ClassLabel(
                num_classes=2, names=['entailment', 'contradiction'], id=None)
            raw_datasets = raw_datasets.cast(
                features)  # overwrite old features
            
        else:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                "glue",
                task_name,
                cache_dir=cache_dir,
                use_auth_token=True if use_auth_token else None,
            )

            if task_name == "qqp":
                # we subsample qqp already here because its really big
                # make sure we fix the seed here
                np.random.seed(123)
                for split in raw_datasets.keys():
                    raw_datasets[split] = raw_datasets[split].select(np.random.choice(
                        np.arange(len(raw_datasets[split])), size=1000, replace=False
                    ))
                    
    # Determine number of labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    return raw_datasets, label_list, num_labels, is_regression


#function almost exact copy from paper's github repo
def load_hans_dataset(cache_dir=None, heuristic=None, subcase=None, label=None):
    # heuristic = {lexical_overlap, subsequence, constituent}
    # subcase = see HANS_SUBCASES
    # label = {0 (entailment), 1 (contradiction)}

    subset = "hans"
    dataset = load_dataset(
        "hans", cache_dir=cache_dir, split="validation")

    # hans comes without indices, so we add them
    indices = list(range(len(dataset)))
    dataset = dataset.add_column(name="idx", column=indices)

    if heuristic is not None:  # filter dataset based on heuristic
        dataset = dataset.filter(
            lambda example: example["heuristic"] == heuristic)
        subset = f"{subset}-{heuristic}"

    if subcase is not None:  # filter dataset based on subcase
        dataset = dataset.filter(
            lambda example: example["subcase"] == subcase)
        subset = f"{subset}-{subcase}"

    if label is not None:  # filter dataset based on label
        dataset = dataset.filter(
            lambda example: example["label"] == label)
        subset = f"{subset}-{'entailment' if label == 0 else 'contradiction'}"

    return dataset, subset

def load_mnli_mismatched_dataset(label=None,cache_dir=None, merge=False):
    subset = "mnli_mm"

    dataset = load_dataset(
        "glue", "mnli", split=f"validation_mismatched", cache_dir=cache_dir)

    if not merge:
        # remove neutral class
        dataset = dataset.filter(
            lambda example: example["label"] != 1)

    # change labels of contradiction examples from 2 to 1
    def change_label(example):
        example["label"] = 1 if example["label"] == 2 else example["label"]
        return example
    dataset = dataset.map(change_label)

    # change features to reflect the new labels
    features = dataset.features.copy()
    features["label"] = ClassLabel(
        num_classes=2, names=['entailment', 'contradiction'], id=None)
    dataset = dataset.cast(
        features)  # overwrite old features

    if label is not None:  # filter dataset based on label
        dataset = dataset.filter(
            lambda example: example["label"] == label)
        subset = f"{subset}-{'entailment' if label == 0 else 'contradiction'}"

    return dataset, subset

def load_paws_qqp_dataset(path, label=None, cache_dir=None):
    # TODO(mm): there's probably a better way of doing this
    data_files = {"validation": path}
    dataset = load_dataset("csv", data_files=data_files,
                           sep="\t", cache_dir=cache_dir)
    dataset = dataset["validation"]

    subset = "paws-qqp"

    def _clean_data(sample):
        # the paws-qqp dataset was created as a stream of bytes. So every sentence starts with "b and ends with ".
        # we remove these
        sample["sentence1"] = sample["sentence1"][2:-1]
        sample["sentence2"] = sample["sentence2"][2:-1]
        return sample

    dataset = dataset.map(_clean_data, batched=False)
    dataset = dataset.rename_column("id", "idx")

    if label is not None:  # filter dataset based on label
        dataset = dataset.filter(
            lambda example: example["label"] == label)
        subset = f"{subset}-{'paraphrase' if label == 1 else 'not-paraphrase'}"

    return dataset, subset

def load_cola_ood_dataset(path, label=None, cache_dir=None):
    # TODO(mm): there's probably a better way of doing this
    data_files = {"validation": path}
    dataset = load_dataset("csv", data_files=data_files, sep="\t", column_names=[
                           'code', 'label', 'annotation', 'sentence'], cache_dir=cache_dir)
    dataset = dataset["validation"]

    # cola-ood comes without indices, so we add them
    indices = list(range(len(dataset)))
    dataset = dataset.add_column(name="idx", column=indices)

    subset = "cola-ood"

    if label is not None:  # filter dataset based on label
        dataset = dataset.filter(
            lambda example: example["label"] == label)
        subset = f"{subset}-{'acceptable' if label == 1 else 'unacceptable'}"

    return dataset, subset



def load_ood_eval_datasets():
        out_of_domain_eval_datasets = {}

        for heuristic in ["lexical_overlap"]:
                for label in [0, 1]:
                        dataset, subset = load_hans_dataset(
                                cache_dir="data", heuristic=heuristic, subcase=None, label=label)
                        print(f"{subset}: {len(dataset)} examples")
                        out_of_domain_eval_datasets[subset] = dataset

        for label in [0, 1]:
                mnli_mm_subset, subset_name = load_mnli_mismatched_dataset(label=label)
                out_of_domain_eval_datasets[subset_name] = mnli_mm_subset

        for label in [0, 1]:
                paws_qqp_subset, subset_name = load_paws_qqp_dataset(
                path=os.path.abspath("./data/paws_qqp/dev_and_test.tsv"), label=label)
                out_of_domain_eval_datasets[subset_name] = paws_qqp_subset

        for label in [0, 1]:
                cola_ood_subset, subset_name = load_cola_ood_dataset(
                path=os.path.abspath("./data/cola_ood/dev.tsv"), label=label)
                out_of_domain_eval_datasets[subset_name] = cola_ood_subset
        return out_of_domain_eval_datasets