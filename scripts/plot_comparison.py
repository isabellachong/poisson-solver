#!/usr/bin/env python3
"""Compare matched serial, MPI CPU, MPI+CUDA and GPU-sharing measurements.

Writes runtime and serial-relative acceleration PNGs for each overlap setting,
plus comparison_summary.csv. Only measured configurations are included. When
the same GPU configuration appears in strong and sharing inputs, use its strong
measurements rather than pooling the two independent sets of repeats.
"""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics


def summarize(filenames, *, size2d=4096, size3d=256, iterations=1000,
              repeats=5, problem='mixed', communication='cuda-aware', overlaps=(0, 1)):
    groups = defaultdict(list)
    protocols = set()
    for filename in filenames:
        with Path(filename).open(newline='') as stream:
            records = list(csv.DictReader(stream))
        if not records:
            raise ValueError(f'Empty input: {filename}')
        for row in records:
            mode, experiment = row['mode'], row['experiment']
            if ((mode == 'serial' and experiment != 'serial')
                    or (mode == 'mpi-cpu' and experiment != 'strong')
                    or (mode == 'mpi-cuda' and experiment not in ('strong', 'sharing'))
                    or mode not in ('serial', 'mpi-cpu', 'mpi-cuda')):
                raise ValueError('Supply serial, CPU/GPU strong-scaling or GPU-sharing CSVs; not weak scaling')
            dim = int(row['dimension'])
            if dim not in (2, 3):
                raise ValueError('Expected dimension 2 or 3')
            shape = tuple(int(row['n'+axis]) for axis in 'xyz')
            target = (size2d, size2d, 1) if dim == 2 else (size3d,)*3
            if shape != target:
                continue
            overlap = int(row['overlap'])
            if mode != 'serial' and overlap not in overlaps:
                continue
            if (row['problem'] != problem or int(row['iterations']) != iterations
                    or int(row['check_every']) != 0):
                raise ValueError('Keep problem and iteration count matched; use fixed iterations')
            protocols.add((row['problem'], int(row['iterations']), int(row['warmup'])))
            ranks, gpus = int(row['mpi_ranks']), int(row['gpus_used'])
            grid = tuple(int(row['p'+axis]) for axis in 'xyz')
            if min(grid) < 1 or math.prod(grid) != ranks:
                raise ValueError('Process grid does not match rank count')
            if int(row['unknowns']) != math.prod(shape):
                raise ValueError('Unknown count does not match the global grid')
            if mode == 'serial':
                if ranks != 1 or gpus != 0 or overlap != 0 or row['communication'] != 'none':
                    raise ValueError('Expected one-core serial baseline without MPI communication')
            elif mode == 'mpi-cpu':
                if ranks < 1 or gpus != 0 or row['communication'] != 'host':
                    raise ValueError('Expected MPI CPU measurements with host communication')
            elif gpus < 1 or ranks < gpus or row['communication'] != communication:
                raise ValueError('GPU count or communication method does not match the comparison')
            seconds = float(row['solve_seconds'])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('Solve times must be finite and positive')
            groups[(dim, mode, ranks, gpus, overlap, experiment)].append(row)
    if len(protocols) != 1:
        raise ValueError('No matching measurements, or mixed problem/iteration/warm-up settings')
    gpu_blocks = {int(r['block_size']) for rows in groups.values()
                  for r in rows if r['mode'] == 'mpi-cuda'}
    if len(gpu_blocks) != 1:
        raise ValueError('Supply GPU measurements with one consistent CUDA block size')
    medians, group_settings = {}, {}
    for key, rows in groups.items():
        if (len(rows) != repeats
                or {int(r['repeat']) for r in rows} != set(range(repeats))):
            raise ValueError(f'Expected {repeats} distinct repeats numbered 0..{repeats-1}: {key}')
        settings = {(r['px'], r['py'], r['pz'], r['communication'], r['block_size']) for r in rows}
        if len(settings) != 1:
            raise ValueError('Process grid, communication or CUDA block size changed within repeats')
        group_settings[key] = settings
        times = [float(r['solve_seconds']) for r in rows]
        medians[key] = (statistics.median(times), min(times), max(times))
    selected = {}
    for key in medians:
        configuration = key[:5]
        previous = selected.get(configuration)
        if previous is not None:
            if group_settings[key] != group_settings[previous]:
                raise ValueError('Strong/sharing settings differ for the same displayed GPU configuration')
            if key[-1] != 'strong':
                continue
        selected[configuration] = key
    summaries = []
    order = {'serial': 0, 'mpi-cpu': 1, 'mpi-cuda': 2}
    for overlap in overlaps:
        for dim in (2, 3):
            baseline_key = selected.get((dim, 'serial', 1, 0, 0))
            if baseline_key is None:
                raise ValueError(f'Missing matching {dim}D serial baseline')
            baseline = medians[baseline_key]
            keys = [k for k in selected if k[0] == dim and (k[1] == 'serial' or k[4] == overlap)]
            if {k[1] for k in keys} != set(order):
                raise ValueError(f'Need serial, MPI CPU and MPI+CUDA results for {dim}D overlap={overlap}')
            for key in sorted(keys, key=lambda k: (order[k[1]], k[3], k[2])):
                _, mode, ranks, gpus, _ = key
                source_key = selected[key]
                sample = groups[source_key][0]
                median, low, high = medians[source_key]
                speedup = baseline[0]/median
                summaries.append(dict(
                    dimension=dim, nx=int(sample['nx']), ny=int(sample['ny']), nz=int(sample['nz']),
                    mode=mode, mpi_ranks=ranks, gpus_used=gpus,
                    source_experiment=source_key[-1],
                    overlap='not applicable' if mode == 'serial' else overlap,
                    comparison_overlap=overlap, communication=sample['communication'],
                    problem=problem, iterations=iterations, warmup=int(sample['warmup']), repeats=repeats,
                    median_seconds=median, min_seconds=low, max_seconds=high,
                    serial_median_seconds=baseline[0], speedup_vs_serial=speedup,
                    cpu_efficiency_vs_serial=speedup/ranks if mode == 'mpi-cpu' else ''))
    return summaries


def label(row):
    if row['mode'] == 'serial':
        return 'Serial\n1 core'
    if row['mode'] == 'mpi-cpu':
        return f'MPI CPU\n{row["mpi_ranks"]} core(s)'
    return f'{row["gpus_used"]} GPU(s)\n{row["mpi_ranks"]} ranks'


def render(summaries, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for overlap in sorted({r['comparison_overlap'] for r in summaries}):
        for metric in ('runtime', 'speedup'):
            fig, axes = plt.subplots(2, 1, figsize=(8.0, 9.0), sharey=True)
            for ax, dim, color, marker in zip(axes, (2, 3), ('#176b98', '#c46b23'), ('o', 's')):
                rows = [r for r in summaries if r['dimension'] == dim and r['comparison_overlap'] == overlap]
                x = list(range(len(rows)))
                if metric == 'runtime':
                    y = [r['median_seconds'] for r in rows]
                    errors = [[r['median_seconds']-r['min_seconds'] for r in rows],
                              [r['max_seconds']-r['median_seconds'] for r in rows]]
                    ax.errorbar(x, y, yerr=errors, fmt=marker, color=color, capsize=3)
                    ax.axhline(rows[0]['serial_median_seconds'], color='#666666', linestyle=':', alpha=.8)
                else:
                    ax.plot(x, [r['speedup_vs_serial'] for r in rows], marker, color=color)
                    ax.axhline(1, color='#666666', linestyle=':', alpha=.8)
                ax.set_xticks(x, [label(r) for r in rows], rotation=45, ha='right', fontsize=8)
                gpu_start = next(i for i, r in enumerate(rows) if r['mode'] == 'mpi-cuda')
                ax.axvline(gpu_start-.5, color='#bbbbbb', linewidth=.8)
                shape = ' x '.join(str(rows[0]['n'+a]) for a in 'xyz'[:dim])
                ax.set(title=f'{dim}D: {shape}', xlabel='Execution configuration',
                       ylabel='Solve time (s)' if metric == 'runtime' else 'Acceleration relative to serial')
                ax.grid(True, axis='y', which='both', alpha=.18)
                ax.tick_params(axis='y', labelleft=True)
            # Set the shared scale after both panels and their error bars exist,
            # so limits include the complete 2D and 3D measurement ranges.
            if metric == 'runtime':
                axes[0].set_yscale('log')
            else:
                axes[0].set_ylim(bottom=0)
            first = summaries[0]
            fig.suptitle(f'Serial / MPI CPU / MPI+CUDA: {first["iterations"]:,} iterations | '
                         f'{first["problem"]} problem\nCommunication overlap {"on" if overlap else "off"} for parallel runs')
            fig.tight_layout(rect=(0, .075, 1, .95), h_pad=2.4)
            note = (f'Median of {first["repeats"]} runs; bars show min-max runtime. Dotted: serial baseline.'
                    if metric == 'runtime' else 'Ratios of median runtimes; dotted: serial baseline (1).')
            fig.text(.5, .025, note+'\nGPU labels specify MPI ranks; CPU labels assume one allocated core per rank.',
                     ha='center', fontsize=9)
            fig.savefig(output/f'comparison_{metric}_overlap{overlap}.png', dpi=300)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--size-2d', type=int, default=4096)
    parser.add_argument('--size-3d', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--problem', default='mixed')
    parser.add_argument('--communication', choices=('host', 'cuda-aware'), default='cuda-aware')
    parser.add_argument('--overlap', nargs='+', type=int, choices=(0, 1), default=[0, 1])
    args = parser.parse_args()
    if min(args.size_2d, args.size_3d, args.iterations, args.repeats) < 1:
        parser.error('Sizes, iterations and repeats must be positive')
    rows = summarize(args.input, size2d=args.size_2d, size3d=args.size_3d,
                     iterations=args.iterations, repeats=args.repeats, problem=args.problem,
                     communication=args.communication, overlaps=sorted(set(args.overlap)))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    render(rows, output)
    with (output/'comparison_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved matched serial/CPU/GPU comparison figures and {len(rows)} summary rows to {output}')


if __name__ == '__main__':
    main()
