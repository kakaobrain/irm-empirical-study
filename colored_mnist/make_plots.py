"""
Make plots for Figures 1 and 3.
"""

import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path

from main import run_colored_mnist

DEFAULT_ARGS = dict(
    hidden_dim=256,
    l2_regularizer_weight=0.001,
    lr=0.001,
    n_restarts=10,
    penalty_anneal_iters=100,
    penalty_weight=10000.0,
    steps=501,
    grayscale_model=False,
    train_probs=(0.2, 0.1),
    train_label_probs=(0.25, ),
    test_prob=0.9,
    n_envs=2,
    n_classes=2,
    verbose=False,
)


def run_color_gap_experiment(color_gap=0.1, erm=False):
    """Q1: Run the default IRM/ERM with a gap parameter."""
    args = copy.deepcopy(DEFAULT_ARGS)
    train1_prob = args['train_probs'][1]
    args['train_probs'] = (
        train1_prob + color_gap / 2,
        train1_prob - color_gap / 2,
    )
    if erm:
        args['penalty_anneal_iters'] = 0
        args['penalty_weight'] = 0.0

    train_accs, test_accs = run_colored_mnist(**args)
    return train_accs, test_accs


def run_label_gap_experiment(label_gap=0.1, erm=False):
    """Q3: Run the default IRM/ERM with a gap parameter for label corruption."""
    args = copy.deepcopy(DEFAULT_ARGS)
    mean_label_prob = args['train_label_probs'][0]
    # test_label_prob is mean of train_label_probs
    args['train_label_probs'] = (
        mean_label_prob + label_gap / 2,
        mean_label_prob - label_gap / 2
    )
    if erm:
        args['penalty_anneal_iters'] = 0
        args['penalty_weight'] = 0.0

    train_accs, test_accs = run_colored_mnist(**args)
    return train_accs, test_accs


def compute_stats(accs, split='train'):
    """Compute mean/std from each trial per gap."""
    index = 0 if split == 'train' else 1
    return zip(*[(np.mean(_accs[index]), np.std(_accs[index]))
                 for _accs in accs])


def run_q1(results_file="q1_results.pkl"):
    """Run Q1 experiment (skip if results_file exists) and make plots."""

    # Run/load results
    gaps = np.linspace(0.0, 0.4, 41)
    print("color gaps tested:", gaps)
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            erm_accs, irm_accs = pickle.load(f)
    else:
        # Main experiment: takes a while
        erm_accs = [run_color_gap_experiment(gap, erm=True) for gap in gaps]
        irm_accs = [run_color_gap_experiment(gap, erm=False) for gap in gaps]
        with open(results_file, "wb") as f:
            pickle.dump((erm_accs, irm_accs), f)

    # Compute mean/std
    erm_train, erm_train_se = compute_stats(erm_accs, 'train')
    erm_test, erm_test_se = compute_stats(erm_accs, 'test')
    irm_train, irm_train_se = compute_stats(irm_accs, 'train')
    irm_test, irm_test_se = compute_stats(irm_accs, 'test')

    # Plot (two-column)
    def _plot(style='seaborn'):
        plt.style.use(style)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        # ax[0]: train
        ax[0].errorbar(gaps, erm_train, yerr=erm_train_se, label='ERM')
        ax[0].errorbar(gaps, irm_train, yerr=irm_train_se, label='IRMv1')
        ax[0].set_title("Train ($p_1 = 0.2 + gap/2$, $p_2 = 0.2 - gap/2$)")
        ax[0].set_xlabel("Probability gap between training environments ($|p_1 - p_2|$)")
        ax[0].set_ylabel("Accuracy")
        ax[0].set_ylim((0.0, 1.0))
        ax[0].legend(loc=4)
        # ax[1]: test
        ax[1].errorbar(gaps, erm_test, yerr=erm_test_se, label='ERM')
        ax[1].errorbar(gaps, irm_test, yerr=irm_test_se, label='IRMv1')
        ax[1].set_title("Test ($p_\mathrm{test} = 0.9$)")
        ax[1].set_xlabel("Probability gap between training environments ($|p_1 - p_2|$)")
        ax[1].set_ylabel("Accuracy")
        ax[1].set_ylim((0.0, 1.0))
        ax[1].legend(loc=4)
        plt.savefig(f"q1_plot_{style}.png", bbox_inches='tight')

    _plot('seaborn')
    _plot('seaborn-colorblind')


def run_q3(results_file="q3_results.pkl"):
    """Run Q3 experiment (skip if results_file exists) and make plots."""

    # Run/load results
    gaps = np.linspace(0.0, 0.5, 51)
    print("label gaps tested:", gaps)
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            erm_accs, irm_accs = pickle.load(f)
    else:
        # Main experiment: takes a while
        erm_accs = [run_label_gap_experiment(gap, erm=True) for gap in gaps]
        irm_accs = [run_label_gap_experiment(gap, erm=False) for gap in gaps]
        with open(results_file, "wb") as f:
            pickle.dump((erm_accs, irm_accs), f)

    # Compute mean/std
    erm_train, erm_train_se = compute_stats(erm_accs, 'train')
    erm_test, erm_test_se = compute_stats(erm_accs, 'test')
    irm_train, irm_train_se = compute_stats(irm_accs, 'train')
    irm_test, irm_test_se = compute_stats(irm_accs, 'test')

    # Plot (two-column)
    def _plot(style='seaborn'):
        plt.style.use(style)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        # ax[0]: train
        ax[0].errorbar(gaps, erm_train, yerr=erm_train_se, label='ERM')
        ax[0].errorbar(gaps, irm_train, yerr=irm_train_se, label='IRMv1')
        ax[0].plot(gaps, [0.5 for _ in gaps], color='orange', linestyle='-', label='Random')
        ax[0].axvline(0.1, color='gray', linestyle='--', label='$|p_1 - p_2|$')
        ax[0].set_title("Train ($\eta_1 = 0.25 + gap/2$, $\eta_2 = 0.25 - gap/2$)")
        ax[0].set_xlabel("Label corruption gap between training environments ($|\eta_1 - \eta_2|$)")
        ax[0].set_ylabel("Accuracy")
        ax[0].set_ylim((0.0, 1.0))
        ax[0].legend(loc=4)
        # ax[1]: test
        ax[1].errorbar(gaps, erm_test, yerr=erm_test_se, label='ERM')
        ax[1].errorbar(gaps, irm_test, yerr=irm_test_se, label='IRMv1')
        ax[1].plot(gaps, [0.5 for _ in gaps], color='orange', linestyle='-', label='Random')
        ax[1].axvline(0.1, color='gray', linestyle='--', label='$|p_1 - p_2|$')
        ax[1].set_title("Test ($\eta_\mathrm{test} = 0.25$)")
        ax[1].set_xlabel("Label corruption gap between training environments ($|\eta_1 - \eta_2|$)")
        ax[1].set_ylabel("Accuracy")
        ax[1].set_ylim((0.0, 1.0))
        ax[1].legend(loc=1)
        plt.savefig(f"q3_plot_{style}.png", bbox_inches='tight')

    _plot('seaborn')
    _plot('seaborn-colorblind')


def main():
    parser = argparse.ArgumentParser(description='Make plots for Q1 and Q3')
    parser.add_argument('--experiment', type=str, choices={'q1', 'q3'})
    args = parser.parse_args()

    if args.experiment == 'q1':
        run_q1()
    else:
        run_q3()


if __name__ == '__main__':
    main()
