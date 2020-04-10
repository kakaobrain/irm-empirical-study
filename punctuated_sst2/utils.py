"""
Utility functions for SST-2
"""

import csv
import os.path


def read_raw_data(path, input_cols, label_col, label_map=None):
    """Read columns from a raw tsv file."""
    with open(path) as f:
        headers = next(f).strip().split("\t")
        inputs, labels = [], []
        for line in f:
            items = line.strip().split("\t")
            inp = tuple(items[headers.index(input_col)]
                        for input_col in input_cols)
            label = items[headers.index(label_col)]
            if label_map is not None:
                label = label_map[label]
            inputs.append(inp)
            labels.append(label)
    return inputs, labels


def write_processed_data(outputs, destdir):
    """Write processed data (one tsv per env)."""
    os.makedirs(destdir, exist_ok=True)
    for name, (inputs, labels) in outputs.items():
        fname = os.path.join(destdir, f"{name}.tsv")
        with open(fname, "w", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter="\t", quotechar=None)
            writer.writerow(["sentence", "label"])
            for inp, label in zip(inputs, labels):
                writer.writerow(list(inp) + [label])
        print("| wrote {} lines to {}".format(
            len(inputs), os.path.join(destdir, name))
        )


def read_processed_data(fname):
    """Read processed data as a list of dictionaries.

    Reads from TSV lines with the following header line:
    sentence    label
    """
    examples = []
    with open(fname, encoding="utf-8") as f:
        for (i, line) in enumerate(f):
            if i == 0:
                continue
            text, label = line.split("\t")
            examples.append({'text': text, 'label': label})
    return examples

