"""
Minimal IRM for ColoredSST-2 with a bag-of-words model
"""

import argparse
import itertools as it
from typing import List

import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import torchtext

from data_processors import get_train_examples, get_test_examples



class BOWClassifier(nn.Module):
    """Simple bag-of-words embeddings + MLP."""
    def __init__(
            self,
            embeddings: torch.FloatTensor,
            n_layers: int,
            n_classes: int,
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(embeddings,
                                                         freeze=True,
                                                         mode='mean')
        self.hidden_dim = self.embedding.embedding_dim
        self.n_layers = n_layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(n_layers - 1)
        ])
        self.n_classes = n_classes
        self.output_layer = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(
            self,
            text: torch.LongTensor,
            offsets: torch.LongTensor,
    ):
        hidden = self.embedding(text, offsets)
        for hidden_layer in self.hidden_layers:
            hidden = F.relu(hidden_layer(hidden))
        return self.output_layer(hidden)


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


def convert_examples_to_features(
        examples: List[dict],
        vocab: torchtext.vocab.Vectors,
        device: torch.device
):
    """Convert examples to torch.Tensors of (text, offsets, labels)."""
    text, offsets, labels = [], [], []
    current_offset = 0
    for example in examples:
        # input
        words = example['text'].split()
        word_ids = [vocab.stoi[word] for word in words if word in vocab.stoi]
        if len(word_ids) < 1:
            continue
        text.extend(word_ids)
        offsets.append(current_offset)
        current_offset += len(word_ids)
        # label
        labels.append(int(example['label']))
    return {
        'text': torch.tensor(text).to(device),
        'offsets': torch.tensor(offsets).to(device),
        'labels': torch.tensor(labels).to(device),
    }


def run_punctuated_sst2(
        datadir: str,
        glove_name: str = "6B",  # 6B, 42B, 840B, twitter.27B
        n_layers: int = 3,
        l2_regularizer_weight: float = 0.001,
        lr: float = 0.001,
        n_restarts: int = 5,
        penalty_anneal_iters: int = 100,
        penalty_weight: float = 10000.0,
        steps: int = 501,
        track_best: bool = False,
        verbose: bool = False
):
    """Run PunctuatedSST-2 experiment and return train/test accuracies."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocab (a torchtext.vocab.Vectors object)
    vocab = torchtext.vocab.GloVe(name=glove_name, dim=300)
    embeddings = vocab.vectors

    # Prepare environments
    train_examples = get_train_examples(datadir)
    test_examples = get_test_examples(datadir)
    n_classes = 2
    train_envs = {env_name: convert_examples_to_features(examples, vocab, device)
                  for env_name, examples in train_examples.items()}
    test_envs = {env_name: convert_examples_to_features(examples, vocab, device)
                 for env_name, examples in test_examples.items()}
    all_envs = [env_name for env_name in it.chain(train_envs, test_envs)]


    final_accs = {env_name: [] for env_name in all_envs}
    best = [{'step': 0, 'min_acc': 0.0, 'loss': 0.0}
            for _ in range(n_restarts)]
    for restart in range(n_restarts):

        # Initialize model
        model = BOWClassifier(embeddings, n_layers, n_classes).to(device)
        if verbose and restart == 0:
            print(model)
            print("# trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Train loop
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if verbose:
            pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test0 acc', 'test1 acc', 'test_ood acc')

        best_min_acc, best_loss, best_step = 0.0, 0.0, 0
        for step in range(steps):
            for _, env in it.chain(train_envs.items(), test_envs.items()):
                logits = model(env['text'], env['offsets'])  # multi-class logit
                env['nll'] = mean_nll(logits, env['labels'])
                env['acc'] = mean_accuracy(logits, env['labels'])
                env['penalty'] = penalty(logits, env['labels'])

            train_nll = torch.stack([env['nll'] for _, env in train_envs.items()]).mean()
            train_acc = torch.stack([env['acc'] for _, env in train_envs.items()]).mean()
            train_penalty = torch.stack([env['penalty'] for _, env in train_envs.items()]).mean()

            weight_norm = torch.tensor(0.).cuda()
            for w in model.parameters():
                if w.requires_grad:
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

            # monitor stats at min_acc for hyperopt (best or last)
            min_acc = min(env['acc'].item() for _, env in test_envs.items())
            if not track_best or min_acc > best_min_acc:
                best_min_acc = min_acc
                best_loss = loss.item()
                best_step = step
                best[restart].update({
                    'step': step,
                    'min_acc': best_min_acc,  # minimum of test accuracies
                    'loss': best_loss,  # training loss
                    'train0_acc': train_envs['train0']['acc'].item(),
                    'train1_acc': train_envs['train1']['acc'].item(),
                    'test0_acc': test_envs['test0']['acc'].item(),
                    'test1_acc': test_envs['test1']['acc'].item(),
                    'test_ood_acc': test_envs['test_ood']['acc'].item(),
                })

            if verbose and step % 100 == 0:
                pretty_print(
                    np.int32(step),
                    train_nll.detach().cpu().numpy(),
                    train_acc.detach().cpu().numpy(),
                    train_penalty.detach().cpu().numpy(),
                    test_envs['test0']['acc'].detach().cpu().numpy(),
                    test_envs['test1']['acc'].detach().cpu().numpy(),
                    test_envs['test_ood']['acc'].detach().cpu().numpy(),
                )

        for env_name in train_envs:
            final_accs[env_name].append(train_envs[env_name]['acc'].item())
        for env_name in test_envs:
            final_accs[env_name].append(test_envs[env_name]['acc'].item())
        if verbose:
            accs = ", ".join(f"{env_name} {best[restart][f'{env_name}_acc']:.5f}"
                             for env_name in all_envs)
            print(f'Restart {restart}: {accs}, '
                  f"min test acc {best[restart]['min_acc']:.5f} (step {best_step})")

    print(f'[Accuracies at best minimum test set accuracy over {n_restarts} restarts]')
    pretty_print("env_name", "mean", "std")
    for env_name in all_envs:
        best_accs = [best[restart][f'{env_name}_acc'] for restart in range(n_restarts)]
        mean, std = np.mean(best_accs), np.std(best_accs)
        pretty_print(env_name, mean, std)
    best_or_last = "Best" if track_best else "Final"
    print(f'[{best_or_last} minimum test set accuracy over {n_restarts} restarts]')
    best_min_accs = [best[restart]['min_acc'] for restart in range(n_restarts)]
    mean, std = np.mean(best_min_accs), np.std(best_min_accs)
    pretty_print("mean", "std")
    pretty_print(mean, std)
    return best


def main():
    parser = argparse.ArgumentParser(description='Minimal PunctuatedSST-2')
    parser.add_argument('--datadir', type=str, default='data/PunctuatedSST-2',
                        help='directory containing PunctuatedSST-2 datasets')
    parser.add_argument('--glove_name', type=str, default='6B',
                        help='name specifying GloVe vectors (default: 6B)')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_restarts', type=int, default=50)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=501)
    parser.add_argument('--track_best', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    run_punctuated_sst2(**vars(args))


if __name__ == '__main__':
    main()
