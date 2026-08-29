"""Report extrema and plot weight distributions for a Qwen checkpoint.

Examples
--------
Analyse the standard Qwen3-1.7B checkpoint::

    HF_HOME=/home/pcs5060ti/Desktop/hf HF_HUB_CACHE=/home/pcs5060ti/Desktop/hf/hub \
    python tools/analyze_weight_distribution.py --model_id Qwen/Qwen3-1.7B

Analyse a local checkpoint, limiting the report to attention projections::

    python tools/analyze_weight_distribution.py \
        --model_id /path/to/Qwen3-1.7B --layer_pattern self_attn

The tool writes a per-tensor CSV, a text summary, and a PNG containing a sampled
global histogram plus distributions of per-tensor minima and maxima.  Sampling makes
the global histogram practical for multi-billion-parameter checkpoints without
materialising every weight on the CPU at once.
"""
import argparse
import os
import re

import torch


def iter_weight_tensors(model, layer_pattern=None, regex=False):
    """Yield floating-point model parameters selected by an optional name filter."""
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        if layer_pattern:
            matched = bool(re.search(layer_pattern, name)) if regex else layer_pattern in name
            if not matched:
                continue
        yield name, parameter


def sample_tensor(tensor, sample_size, generator):
    """Return at most ``sample_size`` values from a tensor on CPU as float32."""
    flat = tensor.detach().float().reshape(-1).cpu()
    if flat.numel() <= sample_size:
        return flat
    # Sampling with replacement keeps temporary memory bounded even for tensors with
    # billions of values; randperm(flat.numel()) would itself be prohibitively large.
    indices = torch.randint(flat.numel(), (sample_size,), generator=generator)
    return flat[indices]


def analyze_tensors(named_tensors, samples_per_tensor=20_000, seed=0):
    """Compute extrema and moments, retaining a bounded representative sample."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    rows = []
    samples = []
    global_min = float('inf')
    global_max = float('-inf')
    total_parameters = 0

    for name, tensor in named_tensors:
        values = tensor.detach().float().reshape(-1).cpu()
        if not values.numel():
            continue
        minimum = float(values.min().item())
        maximum = float(values.max().item())
        sampled = sample_tensor(values, samples_per_tensor, generator)
        rows.append({
            'tensor': name,
            'shape': tuple(tensor.shape),
            'numel': values.numel(),
            'min': minimum,
            'max': maximum,
            'mean': float(values.mean().item()),
            'std': float(values.std(unbiased=False).item()),
            'zero_fraction': float((values == 0).float().mean().item()),
        })
        samples.append(sampled)
        global_min = min(global_min, minimum)
        global_max = max(global_max, maximum)
        total_parameters += values.numel()

    if not rows:
        raise RuntimeError('No floating-point tensors matched --layer_pattern.')
    return rows, torch.cat(samples), global_min, global_max, total_parameters


def plot_distributions(samples, table, output_path, bins):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(samples, bins=bins, color='steelblue', edgecolor='none')
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_title('Sampled weight distribution')
    axes[0].set_xlabel('weight value')
    axes[0].set_ylabel('sample count')

    axes[1].hist(table['min'], bins=bins, color='tomato', edgecolor='none')
    axes[1].set_title('Per-tensor minimum')
    axes[1].set_xlabel('minimum weight value')

    axes[2].hist(table['max'], bins=bins, color='seagreen', edgecolor='none')
    axes[2].set_title('Per-tensor maximum')
    axes[2].set_xlabel('maximum weight value')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def load_model(model_id, dtype, device_map):
    from model_loading_utils import load_config_with_remote_code_fallback, _resolve_model_class

    config, trust_remote_code = load_config_with_remote_code_fallback(model_id)
    model_class = _resolve_model_class(model_id, config)
    print(f'Loading {model_id} with {model_class.__name__} ...')
    return model_class.from_pretrained(
        model_id, dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code)


def run_analysis(model, layer_pattern, regex, samples_per_tensor, seed, bins, results_dir):
    import pandas as pd

    rows, samples, global_min, global_max, total_parameters = analyze_tensors(
        iter_weight_tensors(model, layer_pattern, regex), samples_per_tensor, seed)
    table = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'weight_distribution_per_tensor.csv')
    plot_path = os.path.join(results_dir, 'weight_distribution.png')
    summary_path = os.path.join(results_dir, 'weight_distribution_summary.txt')
    table.to_csv(csv_path, index=False)
    plot_distributions(samples, table, plot_path, bins)

    summary = (
        f'Tensors analysed: {len(rows)}\n'
        f'Parameters analysed: {total_parameters:,}\n'
        f'Global minimum: {global_min:.8g}\n'
        f'Global maximum: {global_max:.8g}\n'
        f'Histogram samples: {len(samples):,}\n'
    )
    with open(summary_path, 'w', encoding='utf-8') as handle:
        handle.write(summary)
    print(summary, end='')
    print(f'Per-tensor report saved to {csv_path}')
    print(f'Distribution plot saved to {plot_path}')
    return table


def synthetic_self_test():
    tensors = [
        ('first.weight', torch.tensor([[-2.0, 0.0], [1.0, 3.0]])),
        ('second.weight', torch.tensor([[4.0, -5.0]])),
    ]
    rows, samples, global_min, global_max, total = analyze_tensors(tensors, 3, seed=0)
    assert global_min == -5.0
    assert global_max == 4.0
    assert total == 6
    assert len(rows) == 2
    assert len(samples) == 6
    assert rows[0]['zero_fraction'] == 0.25
    print('ALL analyze_weight_distribution.py CHECKS PASSED')


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--synthetic', action='store_true', help='Run no-download checks.')
    parser.add_argument('--model_id', default='Qwen/Qwen3-1.7B',
                        help='Hugging Face model ID or local checkpoint directory.')
    parser.add_argument('--layer_pattern', help='Only include tensor names containing this text.')
    parser.add_argument('--regex_pattern', action='store_true', help='Interpret --layer_pattern as a regex.')
    parser.add_argument('--model_dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--device_map', default='cpu')
    parser.add_argument('--samples_per_tensor', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--results_dir', default='results/weight_distribution')
    return parser.parse_args()


def main():
    args = parse_cli_args()
    if args.synthetic:
        synthetic_self_test()
        return 0
    if args.samples_per_tensor < 1 or args.bins < 1:
        raise ValueError('--samples_per_tensor and --bins must be positive.')
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[args.model_dtype]
    model = load_model(args.model_id, dtype, args.device_map)
    run_analysis(model, args.layer_pattern, args.regex_pattern, args.samples_per_tensor,
                 args.seed, args.bins, args.results_dir)
    return 0


if __name__ == '__main__':
    main()
