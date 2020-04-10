# Copyright (c) Facebook, Inc. and its affiliates and Kakao Brain.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from typing import Union, List, Tuple

import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd


class MLP(nn.Module):
    def __init__(self, hidden_dim, n_classes, grayscale_model=False):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.grayscale_model = grayscale_model
        if self.grayscale_model:
            lin1 = nn.Linear(14 * 14, self.hidden_dim)
        else:
            lin1 = nn.Linear(self.n_classes * 14 * 14, self.hidden_dim)
        lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        lin3 = nn.Linear(self.hidden_dim, self.n_classes)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        if self.grayscale_model:
            out = input.view(input.shape[0], self.n_classes, 14 * 14).sum(dim=1)
        else:
            out = input.view(input.shape[0], self.n_classes * 14 * 14)
        out = self._main(out)
        return out


def make_environment(images, labels, color_prob, label_prob, n_classes=2):
    """Build an environment where the label is spuriously correlated with
    a specific "color" channel w.p. `color_prob`.

    The label is also corrupted w.p. `label_prob`, such that
    "color" is more correlated to the true label during training.

    `n_classes` determines how many label classes are used.
        - one color channel per class is created.
        - label corruption shifts label "to the right":
            0 to 1, 1 to 2, ..., and 9 to 0.
    """
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def collapse_labels(labels, n_classes):
        """Collapse 10 classes into n_classes classes."""
        assert n_classes in [2, 3, 5, 10]
        bin_width = 10 // n_classes
        return (labels / bin_width).clamp(max=n_classes - 1)

    def corrupt(labels, n_classes, prob):
        """Corrupt a fraction of labels by shifting it +1 (mod n_classes),
        according to bernoulli(prob).

        Generalizes torch_xor's role of label flipping for the binary case.
        """
        is_corrupt = torch_bernoulli(prob, len(labels)).bool()
        return torch.where(is_corrupt, (labels + 1) % n_classes, labels)

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a label based on the digit
    labels = collapse_labels(labels, n_classes).float()
    # *Corrupt* label with probability 0.25 (default)
    labels = corrupt(labels, n_classes, label_prob)
    # Assign a color based on the label; flip the color with probability e
    colors = corrupt(labels, n_classes, color_prob)
    # Apply the color to the image by only giving image in the assigned color channel
    n, h, w = images.size()
    colored_images = torch.zeros((n, n_classes, h, w)).to(images)
    colored_images[torch.tensor(range(n)), colors.long(), :, :] = images
    return {
        'images': (colored_images.float() / 255.).cuda(),
        'labels': labels.long().cuda(),
    }


def mean_nll(logits, y):
    return nn.functional.cross_entropy(logits, y)


def mean_accuracy(logits, y):
    preds = torch.argmax(logits, dim=1).float()
    return (preds == y).float().mean()


def penalty(logits, y):
    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


def run_colored_mnist(
        hidden_dim: int = 256,
        l2_regularizer_weight: float = 0.001,
        lr: float = 0.001,
        n_restarts: int = 10,
        penalty_anneal_iters: int = 100,
        penalty_weight: float = 10000.0,
        steps: int = 501,
        grayscale_model: bool = False,
        train_probs: Union[Tuple[float], List[float]] = (0.2, 0.1),
        train_label_probs: Union[Tuple[float], List[float]] = (0.25, ),
        test_prob: float = 0.9,
        n_envs: int = 2,
        n_classes: int = 2,
        verbose: bool = False
):
    """Run ColoredMNIST experiment and return train/test accuracies.

    Some key parameters:
        train_probs: tuple of environment probabilities (p_e)
        test_prob: test environment probability
        train_label_probs: tuple of label corruption (flipping) rates per env
        n_envs: number of training environments
        n_classes: number of output classes
    """

    def format_probs(probs):
        if len(probs) == 1:
            return tuple(probs[0] for _ in range(n_envs))
        elif len(probs) == 2:
            lo, hi = sorted(probs)
            return tuple(np.linspace(hi, lo, n_envs))
        else:
            assert len(probs) == n_envs
            return tuple(float(p) for p in probs)

    train_probs = format_probs(train_probs)
    train_label_probs = format_probs(train_label_probs)

    if verbose:
        print('Flags:')
        for k, v in sorted(locals().items()):
            if not callable(v):
                print("\t{}: {}".format(k, v))

    final_train_accs = []
    final_test_accs = []
    for restart in range(n_restarts):

        # Load MNIST, make train/val splits, and shuffle train set examples
        mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:], mnist.targets[50000:])

        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0].numpy())
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1].numpy())

        # Build environments (last one is always test)
        envs = [
            make_environment(mnist_train[0][i::n_envs],
                             mnist_train[1][i::n_envs],
                             train_probs[i],
                             train_label_probs[i],
                             n_classes)
            for i in range(n_envs)
        ]
        test_label_prob = np.mean(train_label_probs).item()
        envs.append(
            make_environment(mnist_val[0],
                             mnist_val[1],
                             test_prob,
                             test_label_prob,
                             n_classes)
        )

        # Define and instantiate the model
        mlp = MLP(hidden_dim, n_classes, grayscale_model).cuda()
        if verbose and restart == 0:
            print(mlp)
            print("# trainable parameters:", sum(p.numel() for p in mlp.parameters() if p.requires_grad))

        # Train loop
        optimizer = optim.Adam(mlp.parameters(), lr=lr)
        if verbose:
            pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

        for step in range(steps):
            for env in envs:
                logits = mlp(env['images'])  # multi-class logit
                env['nll'] = mean_nll(logits, env['labels'])
                env['acc'] = mean_accuracy(logits, env['labels'])
                env['penalty'] = penalty(logits, env['labels'])

            train_nll = torch.stack([envs[i]['nll'] for i in range(n_envs)]).mean()
            train_acc = torch.stack([envs[i]['acc'] for i in range(n_envs)]).mean()
            train_penalty = torch.stack([envs[i]['penalty'] for i in range(n_envs)]).mean()

            weight_norm = torch.tensor(0.).cuda()
            for w in mlp.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += l2_regularizer_weight * weight_norm
            annealed_penalty_weight = (penalty_weight
                if step >= penalty_anneal_iters else 1.0)
            loss += annealed_penalty_weight * train_penalty
            if annealed_penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= annealed_penalty_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_acc = envs[n_envs]['acc']
            if verbose and step % 100 == 0:
                pretty_print(
                    np.int32(step),
                    train_nll.detach().cpu().numpy(),
                    train_acc.detach().cpu().numpy(),
                    train_penalty.detach().cpu().numpy(),
                    test_acc.detach().cpu().numpy()
                )

        train_acc = train_acc.item()
        test_acc = test_acc.item()
        final_train_accs.append(train_acc)
        final_test_accs.append(test_acc)
        if verbose:
            print(f'Restart {restart}: train {train_acc:.5f}, test {test_acc:.5f}')

    print('Final train accuracy (mean/std across restarts):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test accuracy (mean/std across restarts):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    return final_train_accs, final_test_accs


def main():
    parser = argparse.ArgumentParser(description='Extended ColoredMNIST')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=501)
    parser.add_argument('--grayscale_model', action='store_true')
    parser.add_argument('--train_probs', type=float, nargs='+', default=(0.2, 0.1))
    parser.add_argument('--train_label_probs', type=float, nargs='+', default=(0.25, 0.25))
    parser.add_argument('--test_prob', type=float, default=0.9)
    parser.add_argument('--n_envs', type=int, default=2)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    run_colored_mnist(**vars(args))


if __name__ == '__main__':
    main()
