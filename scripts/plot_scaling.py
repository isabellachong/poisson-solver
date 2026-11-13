#!/usr/bin/env python3
"""Plot benchmark.py CSVs and export median runtime/speedup/efficiency tables.

Requires matplotlib. Blue circles denote 2D and orange squares denote 3D;
overlap settings occupy separate panels. Non-timing axes use ordinary scale
except resource count and problem size, which use logarithmic scale.
Only PNG figures and summary.csv are generated. GPU sharing never produces a GPU
hardware-efficiency value. GPU strong scaling keeps ranks/GPU or total ranks
constant; GPU weak scaling keeps ranks/GPU constant.
Speedup uses the smallest resource count in each series, identified in the table.
CPU MPI figures count one allocated CPU core per rank. Weak-scaling size plots
vary hardware together with problem size; they are not fixed-resource sweeps.
"""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics


COLORS = {'2': '#176b98', '3': '#c46b23'}
MARKERS = {'2': 'o', '3': 's'}
BACKENDS = {'serial': 'Serial CPU', 'mpi-cpu': 'MPI CPU', 'mpi-cuda': 'MPI+CUDA'}
WEAK_SIZE_NOTE = (
    'A separate parallel runtime-versus-problem-size study at fixed core/GPU count '
    'was not measured. Replotting weak-scaling data against problem size is '
    'possible, but it must remain labelled as weak scaling because hardware '
    'also changes.')


def legend_title(curves, derived=False):
    counts = {count for curve in curves.values() for count in curve['counts']}
    if len(counts) == 1:
        return f'{"From median" if derived else "Median"} of {next(iter(counts))} runs'
    return 'From median runtimes' if derived else 'Median; min-max error bars'


def style_axes(ax, xlabel, ylabel, *, log_y=False, resource_points=None):
    # Set logarithmic transforms only after all error-bar series are added.
    ax.set(xscale='log', xlabel=xlabel, ylabel=ylabel)
    if log_y:
        ax.set_yscale('log')
    if resource_points is not None:
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ax.xaxis.set_major_locator(FixedLocator(sorted(resource_points)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='both', alpha=.18)


def plot_scaling_group(plt, output, key, schedules, metric):
    kind, mode, communication, problem, iterations = key
    overlaps = sorted(schedules)
    fig, axes = plt.subplots(1, len(overlaps), squeeze=False,
                             figsize=(5*len(overlaps), 4.8))
    footnotes = []
    labels = {'runtime': 'Solve time (s)', 'speedup': 'Relative speedup',
              'efficiency': 'Weak-scaling efficiency' if kind == 'weak' else 'Parallel efficiency'}
    for ax, overlap in zip(axes[0], overlaps):
        curves = schedules[overlap]
        resources = set()
        baselines = set()
        for dimension, curve in sorted(curves.items()):
            points = curve['points']
            resources.update(points)
            baselines.add(points[0])
            values = curve[metric]
            color, marker = COLORS[dimension], MARKERS[dimension]
            if metric == 'runtime':
                errors = [[t-low for t, low in zip(values, curve['low'])],
                          [high-t for t, high in zip(values, curve['high'])]]
                ax.errorbar(points, values, yerr=errors, marker=marker,
                            markersize=5, capsize=3, color=color, label=f'{dimension}D')
                if kind in ('strong', 'weak') and len(points) > 1:
                    ideal = ([values[0]*points[0]/n for n in points] if kind == 'strong'
                             else [values[0]]*len(points))
                    ax.plot(points, ideal, ':', color=color, alpha=.7)
            else:
                ax.plot(points, values, marker=marker, markersize=5,
                        color=color, label=f'{dimension}D')
                if metric == 'speedup' and kind == 'strong' and len(points) > 1:
                    ax.plot(points, [n/points[0] for n in points], ':', color=color, alpha=.7)
        if metric == 'efficiency':
            ax.axhline(1, linestyle=':', color='#666666', alpha=.8)
        if mode == 'mpi-cpu':
            xlabel, resource_unit = 'CPU count', 'CPU(s)'
        elif kind != 'sharing':
            xlabel, resource_unit = 'GPUs used', 'GPU(s)'
        else:
            # GPU-sharing ranks are processes; the allocated CPU-core count
            # does not change with them and must not be relabelled as CPUs.
            xlabel, resource_unit = 'MPI rank count', 'rank(s)'
        style_axes(ax, xlabel, labels[metric], log_y=metric == 'runtime', resource_points=resources)
        if metric != 'runtime':
            ax.set_ylim(bottom=0)
        ax.set_title(f'Communication overlap {"on" if overlap == "1" else "off"}', fontsize=10)
        ax.legend(title=legend_title(curves, derived=metric != 'runtime'), fontsize=9,
                  title_fontsize=9)
        baseline_text = ', '.join(str(n) for n in sorted(baselines))
        notes = [f'Baseline: {baseline_text} {resource_unit}']
        if kind != 'sharing' and (metric == 'efficiency' or len(resources) > 1):
            notes.append('Dotted: ideal scaling')
        if mode == 'mpi-cuda' and kind != 'sharing' and len(resources) == 1:
            notes.append('Only one GPU count measured; comparison pending')
        footnotes.append('\n'.join(notes))
    gpu_counts = {count for curves in schedules.values() for curve in curves.values()
                  for count in curve['gpus']}
    sharing_label = f'GPU sharing ({next(iter(gpu_counts))} GPU(s) fixed)'
    study = {'strong': 'strong scaling', 'weak': 'weak scaling',
             'sharing': sharing_label}[kind]
    fig.suptitle(f'{BACKENDS[mode]} {study}: {labels[metric].lower()}\n'
                 f'{int(iterations):,} iterations | {problem} problem | {communication} communication',
                 fontsize=12)
    fig.tight_layout(rect=(0, .1, 1, 1))
    for ax, note in zip(axes[0], footnotes):
        position = ax.get_position()
        fig.text((position.x0+position.x1)/2, .015, note, ha='center', va='bottom', fontsize=8)
    stem = f'{kind}_{mode}_{communication}_{problem}_k{iterations}'
    fig.savefig(output/(stem+f'_{metric}.png'), dpi=300)
    plt.close(fig)


def plot_weak_problem_size(plt, output, key, schedules, field):
    _, mode, communication, problem, iterations = key
    overlaps = sorted(schedules)
    fig, axes = plt.subplots(1, len(overlaps), squeeze=False,
                             figsize=(5*len(overlaps), 4.8))
    xlabel = ('Global interior unknown count M' if field == 'unknowns' else
              'Implicit matrix nonzero count')
    resource_unit = 'CPUs' if mode == 'mpi-cpu' else 'GPUs'
    for ax, overlap in zip(axes[0], overlaps):
        curves = schedules[overlap]
        for dimension, curve in sorted(curves.items()):
            times = curve['runtime']
            errors = [[t-low for t, low in zip(times, curve['low'])],
                      [high-t for t, high in zip(times, curve['high'])]]
            resources = curve['points']
            count_label = (str(resources[0]) if resources[0] == resources[-1] else
                           f'{resources[0]}-{resources[-1]}')
            ax.errorbar(curve[field], times, yerr=errors, marker=MARKERS[dimension],
                        markersize=5, capsize=3, color=COLORS[dimension],
                        label=f'{dimension}D ({count_label} {resource_unit})')
        style_axes(ax, xlabel, 'Solve time (s)', log_y=True)
        ax.set_title(f'Communication overlap {"on" if overlap == "1" else "off"}', fontsize=10)
        ax.legend(title=legend_title(curves), fontsize=9, title_fontsize=9)
    quantity = 'unknown count' if field == 'unknowns' else 'matrix nonzeros'
    fig.suptitle(f'{BACKENDS[mode]} weak scaling: runtime versus {quantity}\n'
                 f'{int(iterations):,} iterations | {problem} problem | {communication} communication',
                 fontsize=12)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.text(.5, .015,
             'Weak scaling: hardware increases with problem size; work per CPU/GPU stays fixed.\n'
             'A separate parallel size sweep at fixed core/GPU count was not measured.',
             ha='center', va='bottom', fontsize=8)
    stem = f'weak_{mode}_{communication}_{problem}_k{iterations}'
    fig.savefig(output/(stem+f'_runtime_vs_{field}.png'), dpi=300)
    plt.close(fig)


def plot_serial_group(plt, output, key, schedules):
    _, _, communication, problem, iterations = key
    curves = schedules['0']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for dimension, curve in sorted(curves.items()):
        times = curve['runtime']
        errors = [[t-low for t, low in zip(times, curve['low'])],
                  [high-t for t, high in zip(times, curve['high'])]]
        for ax, field in zip(axes, ('points', 'nonzeros')):
            points = curve[field]
            ax.errorbar(points, times, yerr=errors, marker=MARKERS[dimension],
                        markersize=5, capsize=3, color=COLORS[dimension], label=f'{dimension}D')
            ax.plot(points, [times[0]*n/points[0] for n in points], ':',
                    color=COLORS[dimension], alpha=.7)
    for ax, xlabel in zip(axes, ('Interior unknowns M', 'Implicit matrix nonzeros')):
        style_axes(ax, xlabel, 'Solve time (s)', log_y=True)
    axes[0].legend(title=legend_title(curves), loc='upper left')
    axes[1].text(.04, .95, 'Dotted: linear-work reference', transform=axes[1].transAxes,
                 va='top', fontsize=9)
    fig.suptitle(f'Vanda serial Jacobi: {int(iterations):,} iterations on one CPU core', fontsize=12)
    fig.tight_layout()
    fig.savefig(output/f'serial_serial_{communication}_{problem}_k{iterations}_runtime.png', dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output-dir', default='results/figures')
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
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
    plot_groups = defaultdict(lambda: defaultdict(dict))
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
        if any(not math.isfinite(float(row['solve_seconds'])) or float(row['solve_seconds']) <= 0
               for row in rows):
            raise ValueError('Runtime must be finite and positive')
        speedups = [times[0]/t for t in times]
        efficiencies = ([times[0]/t for t in times] if kind == 'weak' else
                        [s/(n/points[0]) for s, n in zip(speedups, points)])
        for i, n in enumerate(points):
            tables.append(dict(experiment=kind, mode=mode, dimension=dim, overlap=overlap,
                               communication=communication, resource_axis=resource_key, resources=n,
                               baseline_resources=points[0], median_seconds=times[i], min_seconds=low[i],
                               max_seconds=high[i], relative_speedup='' if kind == 'serial' else speedups[i],
                               efficiency=efficiencies[i] if kind in ('strong', 'weak') else ''))
        plot_key = (kind, mode, communication, problem, iterations)
        plot_groups[plot_key][overlap][dim] = dict(
            points=points, runtime=times, low=low, high=high, speedup=speedups,
            efficiency=efficiencies, counts=[len(grouped[n]) for n in points],
            unknowns=[int(grouped[n][0]['unknowns']) for n in points],
            nonzeros=[int(grouped[n][0]['nonzeros']) for n in points],
            gpus=[int(grouped[n][0]['gpus_used']) for n in points])
    for key, schedules in plot_groups.items():
        kind = key[0]
        if kind == 'serial':
            plot_serial_group(plt, output, key, schedules)
            continue
        metrics = ['runtime']
        if kind in ('strong', 'weak'):
            metrics.append('efficiency')
        if kind in ('strong', 'sharing'):
            metrics.append('speedup')
        for metric in metrics:
            plot_scaling_group(plt, output, key, schedules, metric)
        if kind == 'weak':
            for field in ('unknowns', 'nonzeros'):
                plot_weak_problem_size(plt, output, key, schedules, field)
    if tables:
        with (output/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
            writer.writeheader(); writer.writerows(tables)
    if any(key[0] == 'weak' for key in plot_groups):
        print(WEAK_SIZE_NOTE)
    print(f'Saved figures and median summary to {output}')


if __name__ == '__main__':
    main()
