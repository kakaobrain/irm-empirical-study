#!/usr/bin/python3 -u

"""
Create environments for PunctuatedSST-2.

Download and unzip SST-2 data from:
    https://gluebenchmark.com/tasks
"""

import argparse
from collections import Counter
import numpy as np
import os.path
import string

from utils import read_raw_data, write_processed_data


def color_sentence(sents, artifact_type, artifact_label):
    """Color (perturb) sentences according to spurious artifacts.

    Artifact label is assumed to be binary.

    [artifact types]
    default: spurious punctuation (contains period or exclamation mark)
    grayscale: return original sentence (no artifact)
    """
    assert isinstance(sents, tuple)
    assert artifact_type in {"default", "grayscale"}
    assert artifact_label in {0, 1}

    if artifact_type == "grayscale":
        return sents

    color = " ." if artifact_label == 1 else " !"
    out = []
    for sent in sents:
        tokens = sent.split()
        if tokens[-1] in string.punctuation:
            tokens = tokens[:-1]
        out.append(" ".join(tokens) + color)
    return tuple(out)


def split_envs(inputs, labels,
               name="", n_envs=1, match_lengths=False, rng=None):
    """Randomly split inputs and labels into different environments.

    Optionally matches the number of samples in each environment. (for train)
    """
    n = len(inputs)
    if rng is None:
        print("warning: RNG not provided, using unknown random seed")
        rng = np.random.RandomState()

    # randomly split into environments
    env_indices = rng.randint(n_envs, size=n)
    out = {
        f"{name}{env}": {"inputs": [], "labels": []}
        for env in range(n_envs)
    }
    for inp, label, env in zip(inputs, labels, env_indices):
        env_name = f"{name}{env}"
        out[env_name]["inputs"].append(inp)
        out[env_name]["labels"].append(label)

    # match lengths between environments
    if match_lengths:
        maxlen = max(len(ds["inputs"]) for env_name, ds in out.items())
        for env_name, ds in out.items():
            n_extra = maxlen - len(ds["inputs"])
            if n_extra >= 1:
                extra_indices = rng.choice(len(ds["inputs"]), size=n_extra)
                ds["inputs"] += [ds["inputs"][i] for i in extra_indices]
                ds["labels"] += [ds["labels"][i] for i in extra_indices]

    # out: { "nameN": {"inputs": inputsN, "labels": labelsN} }
    return out


def color_binary_dataset(inputs, labels, artifact_type,
                         flip_label=0.25, p_env=0.1, rng=None):
    """Give artifical "color" tokens to inputs that correlate with the label.

    *Assumed: label is binary.*

    Analogous to the colored MNIST dataset construction in the IRM paper."""

    if rng is None:
        print("warning: RNG not provided, using unknown random seed")
        rng = np.random.RandomState()

    colored_inputs, colored_labels = [], []
    for input_sent, label in zip(inputs, labels):
        # randomly flip labels
        if flip_label > 0.0 and rng.random(1).item() < flip_label:
            label = 1 - label
        # assign artifact by environment probability
        artifact_label = label if rng.random(1).item() >= p_env else 1 - label
        colored_inputs.append(
            color_sentence(input_sent, artifact_type, artifact_label)
        )
        colored_labels.append(label)

    return colored_inputs, colored_labels


def color_sst2(datadir, destdir, version):
    """Generate a PunctuatedSST-2 dataset.

    datadir is the location that contains the original SST-2 data."""

    # default setup
    n_envs = 2
    p_env_test = 0.9
    label_map = {"0": 0, "1": 1}
    rng = np.random.RandomState(1)
    if version == "grayscale":
        artifact_type, flip_label, p_envs = ("grayscale", 0.25, [0.0, 0.0])
    else:
        artifact_type, flip_label, p_envs = ("default", 0.25, [0.2, 0.1])

    # train: train0(p=0.2), train1(p=0.1)
    inputs, labels = read_raw_data(
        os.path.join(datadir, "train.tsv"), ["sentence"], "label", label_map
    )
    train = split_envs(
        inputs, labels,
        name="train", n_envs=n_envs, match_lengths=True, rng=rng
    )
    for env in range(n_envs):
        ctr = Counter(train[f"train{env}"]["labels"])
        majority_ratio = ctr.most_common(1)[0][1] / sum(ctr.values())
        print(f"train{env}:", ctr, ", majority:", majority_ratio)
    train0, train1 = [
        color_binary_dataset(train[f"train{env}"]["inputs"],
                             train[f"train{env}"]["labels"],
                             artifact_type,
                             flip_label=flip_label,
                             p_env=p_env,
                             rng=rng)
        for env, p_env in enumerate(p_envs)
    ]

    # test: test0(p=0.2), test1(p=0.1), test_ood(p=0.9)
    # (we use the SST-2 dev set for all evaluation)
    inputs, labels = read_raw_data(
        os.path.join(datadir, "dev.tsv"), ["sentence"], "label", label_map
    )
    test = split_envs(
        inputs, labels,
        name="test", n_envs=n_envs + 1, match_lengths=False, rng=rng
    )
    for env in range(n_envs + 1):
        ctr = Counter(test[f"test{env}"]["labels"])
        majority_ratio = ctr.most_common(1)[0][1] / sum(ctr.values())
        print(f"test{env}:" if env < n_envs else "test_ood", ctr,
              ", majority:", majority_ratio)
    test0, test1, test_ood = [
        color_binary_dataset(test[f"test{env}"]["inputs"],
                             test[f"test{env}"]["labels"],
                             artifact_type,
                             flip_label=flip_label,
                             p_env=p_env,
                             rng=rng)
        for env, p_env in enumerate(p_envs + [p_env_test])
    ]

    outputs = {
        "train0": train0,
        "train1": train1,
        "test0": test0,
        "test1": test1,
        "test_ood": test_ood,
    }

    write_processed_data(outputs, destdir)

    return train0, train1, test0, test1, test_ood


def main():
    parser = argparse.ArgumentParser(description="make environments for PunctuatedSST-2")
    parser.add_argument('--datadir', help="directory containing raw data")
    parser.add_argument('--destdir', help="output directory")
    parser.add_argument('--version', default="default",
                        help="dataset version (default or grayscale)")
    args = parser.parse_args()
    color_sst2(args.datadir, args.destdir, args.version)


if __name__ == "__main__":
    main()
