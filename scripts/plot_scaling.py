#!/usr/bin/env python3
"""Plot benchmark.py CSVs and export median runtime/speedup/efficiency tables.

Requires matplotlib. Non-timing axes use ordinary scale except resource count and
problem size, which use logarithmic scale. GPU sharing never produces a GPU
hardware-efficiency value. GPU strong scaling keeps ranks/GPU or total ranks
constant; GPU weak scaling keeps ranks/GPU constant.
Speedup uses the smallest resource count in each series, identified in the table.
"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output-dir', default='results/figures')
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series = defaultdict(list)
    for filename in a.input:
        with Path(filename).open() as stream:
            for row in csv.DictReader(stream):
                if int(row['check_every']) != 0:
                    raise ValueError('These plots require matched fixed-iteration measurements')
                key = tuple(row[k] for k in ('experiment', 'mode', 'dimension', 'overlap', 'communication', 'problem', 'iterations'))
                series[key].append(row)
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = []
    for key, rows in series.items():
        kind, mode, dim, overlap, communication, problem, iterations = key
        if mode == 'mpi-cuda' and kind in ('strong', 'weak'):
            ratios = {int(r['mpi_ranks'])/int(r['gpus_used']) for r in rows}
            fixed_ranks = len({r['mpi_ranks'] for r in rows}) == 1
            if len(ratios) != 1 and (kind == 'weak' or not fixed_ranks):
                raise ValueError('GPU strong scaling must keep ranks/GPU or total MPI ranks constant; weak scaling keeps ranks/GPU constant. A fixed-GPU rank sweep is a sharing study.')
        if kind == 'sharing' and len({r['gpus_used'] for r in rows}) != 1:
            raise ValueError('Sharing input must hold the GPU count fixed')
        resource_key = 'unknowns' if kind == 'serial' else ('gpus_used' if mode == 'mpi-cuda' and kind != 'sharing' else 'mpi_ranks')
        grouped = defaultdict(list)
        for row in rows:
            grouped[int(row[resource_key])].append(row)
        points = sorted(grouped)
        if kind in ('strong', 'sharing') and len({r['unknowns'] for r in rows}) != 1:
            raise ValueError('Strong scaling and sharing input must hold global unknown count fixed')
        if kind == 'weak':
            workloads = {int(r['unknowns'])/int(r[resource_key]) for r in rows}
            if len(workloads) != 1:
                raise ValueError('Weak scaling input must hold unknowns per resource fixed')
        times = [statistics.median(float(r['solve_seconds']) for r in grouped[n]) for n in points]
        low = [min(float(r['solve_seconds']) for r in grouped[n]) for n in points]
        high = [max(float(r['solve_seconds']) for r in grouped[n]) for n in points]
        if min(times) <= 0:
            raise ValueError('Runtime must be positive')
        speedups = [times[0]/t for t in times]
        efficiencies = ([times[0]/t for t in times] if kind == 'weak' else
                        [s/(n/points[0]) for s, n in zip(speedups, points)])
        stem = f'{kind}_{mode}_{dim}d_overlap{overlap}_{communication}_{problem}_k{iterations}'
        for i, n in enumerate(points):
            tables.append(dict(experiment=kind, mode=mode, dimension=dim, overlap=overlap,
                               communication=communication, resource_axis=resource_key, resources=n,
                               baseline_resources=points[0], median_seconds=times[i], min_seconds=low[i],
                               max_seconds=high[i], relative_speedup='' if kind == 'serial' else speedups[i],
                               efficiency=efficiencies[i] if kind in ('strong', 'weak') else ''))
        xlabel = {'unknowns': 'Global unknown count', 'mpi_ranks': 'MPI rank count', 'gpus_used': 'GPUs used'}[resource_key]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(points, times, yerr=[[t-l for t, l in zip(times, low)], [h-t for t, h in zip(times, high)]], marker='o', capsize=4)
        ax.set(xscale='log', yscale='log', xlabel=xlabel, ylabel='Solve time (s)', title=stem.replace('_', ' '))
        ax.grid(True, which='both', alpha=.25)
        fig.tight_layout(); fig.savefig(output/(stem+'_runtime.png'), dpi=180); plt.close(fig)
        if kind in ('strong', 'weak'):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(points, efficiencies, 'o-')
            ax.set(xscale='log', xlabel=xlabel, ylabel='Parallel efficiency' if kind == 'strong' else 'Weak-scaling efficiency',
                   title=f'Baseline: {points[0]} {xlabel.lower()}')
            ax.grid(True, which='both', alpha=.25)
            fig.tight_layout(); fig.savefig(output/(stem+'_efficiency.png'), dpi=180); plt.close(fig)
        if kind in ('strong', 'sharing'):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(points, speedups, 'o-')
            ax.set(xscale='log', xlabel=xlabel, ylabel='Relative speedup', title=f'Baseline: {points[0]} {xlabel.lower()}')
            ax.grid(True, which='both', alpha=.25)
            fig.tight_layout(); fig.savefig(output/(stem+'_speedup.png'), dpi=180); plt.close(fig)
    if tables:
        with (output/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
            writer.writeheader(); writer.writerows(tables)
    print(f'Saved figures and median summary to {output}')


if __name__ == '__main__':
    main()
