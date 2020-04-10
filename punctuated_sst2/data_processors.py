
"""
Data processors for multi-environment settings.

Reference:
    https://github.com/huggingface/transformers/blob/master/src/transformers/data/processors/glue.py
"""

from glob import glob
import os.path

from utils import read_processed_data


def get_env_name(fname):
    basename = os.path.basename(fname)
    env_name, ext = os.path.splitext(basename)
    assert ext == ".tsv"
    return env_name



def get_train_examples(data_dir):
    """Load training examples from multiple environments."""
    train_envs = glob(os.path.join(data_dir, "train*.tsv"))
    print(f"loading train data from {len(train_envs)} environments")

    # return examples as dict; check that sizes match for training
    train_examples = {}
    prev_length = None
    for train_env in train_envs:
        env_name = get_env_name(train_env)
        examples = read_processed_data(train_env)
        if prev_length:
            assert len(examples) == prev_length, \
                f"data size between training environments differ"

        train_examples[env_name] = examples
        prev_length = len(examples)
    return train_examples


def get_test_examples(data_dir):
    """Load test examples from multiple environments."""
    test_envs = glob(os.path.join(data_dir, "test*.tsv"))
    print(f"loading test data from {len(test_envs)} environments")

    # test0: examples0, test1: examples1, test_ood: examples_ood
    test_examples = {}
    for test_env in test_envs:
        env_name = get_env_name(test_env)
        examples = read_processed_data(test_env)
        test_examples[env_name] = examples
    return test_examples
