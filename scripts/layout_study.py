#!/usr/bin/env python3
"""Supplementary Cartesian layout experiments and their analysis.

Subcommands: benchmark, plot, transport, audit, bind. This consolidates the
report appendix's benchmark_part3, plot_part3, analyze_gpu_part3, audit_part3,
validate_support and run_cpu_bound tools. Raw timings are never synthesized.
"""
import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from benchmark import shape
from solver_output import parse_summary


def validate_record(row, case, mode, dim, iterations):
    expected = dict(mode=mode, dimension=dim, mpi_ranks=math.prod(case['grid']),
                    iterations=iterations, check_every=0, overlap=case['overlap'],
                    communication=case['communication'], problem='mixed', warmup=3)
    expected.update({'n'+a: n for a, n in zip('xyz', case['points'])})
    expected.update({'p'+a: n for a, n in zip('xyz', case['grid'])})
    for key, value in expected.items():
        if row.get(key) != str(value):
            raise ValueError(f'Mismatched {key}: expected {value}, got {row.get(key)}')
    if int(row['unknowns']) != math.prod(case['points']):
        raise ValueError('Incorrect global unknown count')
    for key in ('solve_seconds', 'residual_inf', 'error_inf', 'error_rms'):
        value = float(row[key])
        if not math.isfinite(value) or value < 0 or (key == 'solve_seconds' and value == 0):
            raise ValueError(f'Invalid numeric result: {key}={value}')
    if mode == 'mpi-cuda':
        ranks = math.prod(case['grid'])
        gpus, maximum = int(row['gpus_used']), int(row['max_ranks_per_gpu'])
        if not (1 <= gpus <= ranks and math.ceil(ranks/gpus) <= maximum <= ranks):
            raise ValueError('Invalid GPU mapping counts')


def build_plan(args):
    grids = [shape(text, args.dim) for text in args.process_grids]
    if len(set(grids)) != len(grids):
        raise ValueError('Duplicate process grids')
    if len({math.prod(grid) for grid in grids}) != 1:
        raise ValueError('Layout comparisons require the same MPI rank count')
    points_list = []
    for text in args.problem_sizes:
        points = tuple(map(int, text.split('x')))
        if len(points) != args.dim or min(points) < 1:
            raise ValueError('Problem sizes must contain positive counts for each dimension')
        if points in points_list:
            raise ValueError('Duplicate problem size')
        points_list.append(points)
    if min(args.repeats, args.iterations, args.timeout) < 1:
        raise ValueError('Repeats, iterations and timeout must be positive')
    if args.mode == 'mpi-cpu' and args.communication != ['host']:
        raise ValueError('CPU experiments require --communication host')
    plan = []
    for points in points_list:
        for grid in grids:
            if any(p > n for p, n in zip(grid, points)):
                raise ValueError('Process grid creates an empty subdomain')
        # Reverse the entire matched-case order on alternate repeats.
        configurations = [(g, t, o) for g in grids for t in args.communication for o in (0, 1)]
        for repeat in range(args.repeats):
            ordered = configurations if repeat % 2 == 0 else configurations[::-1]
            for grid, transport, overlap in ordered:
                executable = [str(Path(args.exe).resolve())]
                if getattr(args, 'bind_cpus', False):
                    executable = [sys.executable, str(Path(__file__).resolve()), 'bind', *executable]
                command = [args.launcher, *args.launcher_arg, '-n', str(math.prod(grid)),
                           *executable, '--dim', str(args.dim),
                           '--iterations', str(args.iterations), '--check-every', '0',
                           '--warmup', '3', '--problem', 'mixed', '--overlap', str(overlap),
                           '--communication', transport]
                for axis, count, parts in zip('xyz', points, grid):
                    command += ['--n'+axis, str(count), '--p'+axis, str(parts)]
                plan.append(dict(points=points, grid=grid, communication=transport,
                                 overlap=overlap, repeat=repeat, command=command))
    return plan


def collect_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', required=True)
    p.add_argument('--mode', choices=['mpi-cpu', 'mpi-cuda'], required=True)
    p.add_argument('--dim', type=int, choices=[2, 3], default=2)
    p.add_argument('--process-grids', nargs='+', required=True)
    p.add_argument('--problem-sizes', nargs='+', required=True)
    p.add_argument('--communication', nargs='+', choices=['host', 'cuda-aware'], default=['host'])
    p.add_argument('--iterations', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--launcher', default='mpiexec')
    p.add_argument('--launcher-arg', action='append', default=[])
    p.add_argument('--allocation-label', required=True,
                   help='Describe hardware/allocation; compare only matching hardware')
    p.add_argument('--bind-cpus', action='store_true',
                   help='Bind local CPU ranks using the allocated POISSON_CPUSET list')
    p.add_argument('--expected-gpus', type=int,
                   help='Require this actual GPU count, e.g. 1 for a single-card sharing study')
    p.add_argument('--output')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.bind_cpus and args.mode != 'mpi-cpu':
        p.error('--bind-cpus is only for CPU layout experiments')
    if args.expected_gpus is not None and (args.mode != 'mpi-cuda' or args.expected_gpus < 1):
        p.error('--expected-gpus requires CUDA mode and a positive count')
    if len(set(args.communication)) != len(args.communication):
        p.error('Duplicate communication methods')
    try:
        plan = build_plan(args)
    except ValueError as error:
        p.error(str(error))
    if args.dry_run:
        for case in plan:
            print(json.dumps(case))
        return
    if not args.output:
        p.error('--output is required unless --dry-run is used')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    gpu_mapping = None
    writer = None
    with output.open('x', newline='') as stream:
        for case in plan:
            print(f"points={case['points']} grid={case['grid']} overlap={case['overlap']} "
                  f"transport={case['communication']} repeat={case['repeat']}", file=sys.stderr)
            run = subprocess.run(case['command'], text=True, capture_output=True, timeout=args.timeout)
            print(run.stderr, end='', file=sys.stderr)
            if run.returncode:
                raise RuntimeError(f'Solver failed ({run.returncode}): {run.stdout}')
            row = parse_summary(run.stdout)
            validate_record(row, case, args.mode, args.dim, args.iterations)
            if args.mode == 'mpi-cuda':
                if args.expected_gpus is not None and int(row['gpus_used']) != args.expected_gpus:
                    raise ValueError('Measured GPU count does not match the allocated study')
                mapping = (int(row['gpus_used']), int(row['max_ranks_per_gpu']), row['device_name'])
                if gpu_mapping is not None and mapping != gpu_mapping:
                    raise ValueError('GPU allocation/mapping changed within the matched experiment')
                gpu_mapping = mapping
            record = dict(experiment='layout', allocation=args.allocation_label,
                          repeat=case['repeat'], command_json=json.dumps(case['command']), **row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)
            stream.flush()
    print(f'Saved {output}')


GROUP_KEYS = ('allocation', 'mode', 'dimension', 'nx', 'ny', 'nz', 'mpi_ranks',
              'px', 'py', 'pz', 'gpus_used', 'max_ranks_per_gpu', 'device_name',
              'communication', 'problem', 'iterations', 'check_every', 'warmup', 'block_size')


def summarize_layouts(rows):
    groups = defaultdict(lambda: {0: {}, 1: {}})
    for row in rows:
        if row['experiment'] != 'layout' or int(row['check_every']) != 0:
            raise ValueError('Expected fixed-iteration layout experiments')
        key = tuple(row[k] for k in GROUP_KEYS)
        overlap, repeat = int(row['overlap']), int(row['repeat'])
        seconds = float(row['solve_seconds'])
        if overlap not in (0, 1) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid overlap or runtime')
        if repeat in groups[key][overlap]:
            raise ValueError('Duplicate repeat: use a different allocation label for another session')
        groups[key][overlap][repeat] = seconds
    summary = []
    for key, timings in groups.items():
        if not timings[0] or timings[0].keys() != timings[1].keys():
            raise ValueError('Each layout needs matching overlap=0/1 repeats')
        record = dict(zip(GROUP_KEYS, key))
        for overlap in (0, 1):
            values = list(timings[overlap].values())
            for name, fn in [('median', statistics.median), ('min', min), ('max', max)]:
                record[f'{name}_seconds_overlap{overlap}'] = fn(values)
        record['repeats'] = len(timings[0])
        record['paired_overlap_speedup'] = statistics.median(
            timings[0][r]/timings[1][r] for r in timings[0])
        summary.append(record)
    return sorted(summary, key=lambda r: tuple(r[k] for k in GROUP_KEYS))


def plot_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--overview-sizes', nargs=2, type=int, metavar=('N2D', 'N3D'),
                   help='Also draw a two-column CPU overview at these square/cubic axis lengths')
    args = p.parse_args()
    rows = []
    for filename in args.input:
        with Path(filename).open(newline='') as stream:
            rows.extend(csv.DictReader(stream))
    summary = summarize_layouts(rows)
    if not summary:
        p.error('No measurements found')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'layout_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Only combine process-grid shapes within an otherwise identical experiment.
    panel_keys = tuple(k for k in GROUP_KEYS if k not in ('px', 'py', 'pz'))
    panels = defaultdict(list)
    for record in summary:
        panels[tuple(record[k] for k in panel_keys)].append(record)
    figure_index = []
    for index, records in enumerate(panels.values()):
        # Runtime-only figure: retain the measured points and their full ranges.
        # Paired overlap ratios remain in layout_summary.csv, not a bar panel.
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        positions = list(range(len(records)))
        labels = ['x'.join(r['p'+a] for a in 'xyz'[:int(r['dimension'])]) for r in records]
        for overlap, offset, color, label in [
                (0, -0.08, '#176b98', 'No overlap (O=0)'),
                (1, 0.08, '#c46b23', 'Interior overlap (O=1)')]:
            times = [r[f'median_seconds_overlap{overlap}'] for r in records]
            errors = [[t-r[f'min_seconds_overlap{overlap}'] for t, r in zip(times, records)],
                      [r[f'max_seconds_overlap{overlap}']-t for t, r in zip(times, records)]]
            ax.errorbar([x+offset for x in positions], times, yerr=errors,
                        fmt='o' if overlap == 0 else 's', capsize=4,
                        markersize=6, color=color, label=label)
        ax.set(yscale='log', ylabel='Solve time (s)', xlabel='Process grid',
               xticks=positions, xticklabels=labels)
        ax.set_xlim(-0.4, max(positions)+0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.legend(loc='upper center', bbox_to_anchor=(.5, .82), ncol=2,
                   frameon=False, fontsize=9)
        ax.grid(axis='y', which='both', alpha=.2)
        r = records[0]
        size = ' x '.join(r['n'+a] for a in 'xyz'[:int(r['dimension'])])
        mode = 'CPU MPI' if r['mode'] == 'mpi-cpu' else 'MPI+CUDA'
        resources = f"{r['mpi_ranks']} MPI ranks"
        if r['mode'] == 'mpi-cuda':
            resources = f"{r['gpus_used']} GPU(s), " + resources
        path = ('CPU memory' if r['mode'] == 'mpi-cpu' else
                'Host staging' if r['communication'] == 'host' else 'CUDA-aware MPI')
        fig.suptitle(f"{mode} | {r['dimension']}D grid {size}\n"
                     f"{resources} | {path} | {int(r['iterations']):,} sweeps", fontsize=11)
        fig.text(.5, .015, f"Points: median of {r['repeats']} repeats; vertical ranges: minimum to maximum.",
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .89))
        fig.savefig(output/f'layout_{index:03d}.png', dpi=240)
        figure_index.append(dict(figure=f'layout_{index:03d}.png',
                                 **{k: r[k] for k in panel_keys}))
        plt.close(fig)
    if args.overview_sizes:
        if min(args.overview_sizes) < 1:
            p.error('Overview grid sizes must be positive')
        overview = []
        for dimension, size in zip((2, 3), args.overview_sizes):
            matches = [records for records in panels.values()
                       if records[0]['mode'] == 'mpi-cpu'
                       and int(records[0]['dimension']) == dimension
                       and all(int(records[0]['n'+a]) == size for a in 'xyz'[:dimension])]
            if len(matches) != 1:
                p.error(f'Need exactly one CPU {dimension}D panel at axis length {size}')
            overview.append(matches[0])
        common_keys = ('mode', 'mpi_ranks', 'gpus_used', 'communication', 'problem',
                       'iterations', 'check_every', 'warmup', 'block_size')
        if any(overview[0][0][k] != overview[1][0][k] for k in common_keys):
            p.error('CPU overview panels must use matching resources and solver settings')
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        for ax, records in zip(axes, overview):
            records = sorted(records, key=lambda r: tuple(int(r['p'+a]) for a in 'xyz'))
            positions = list(range(len(records)))
            labels = ['x'.join(r['p'+a] for a in 'xyz'[:int(r['dimension'])]) for r in records]
            handles = []
            for overlap, offset, color, label in [
                    (0, -.08, '#176b98', 'No overlap (O=0)'),
                    (1, .08, '#c46b23', 'Interior overlap (O=1)')]:
                times = [r[f'median_seconds_overlap{overlap}'] for r in records]
                errors = [[t-r[f'min_seconds_overlap{overlap}'] for t, r in zip(times, records)],
                          [r[f'max_seconds_overlap{overlap}']-t for t, r in zip(times, records)]]
                handles.append(ax.errorbar([x+offset for x in positions], times, yerr=errors,
                               fmt='o' if overlap == 0 else 's', color=color,
                               markersize=5, capsize=3, label=label))
            r = records[0]
            size = ' x '.join(r['n'+a] for a in 'xyz'[:int(r['dimension'])])
            ax.set(yscale='log', title=f"{r['dimension']}D: {size}", ylabel='Solve time (s)',
                   xlabel='Process grid', xticks=positions, xticklabels=labels)
            ax.set_xlim(-.4, max(positions)+.4)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=10); ax.grid(axis='y', which='both', alpha=.2)
        r = overview[0][0]
        fig.suptitle(f"CPU MPI | {r['mpi_ranks']} CPU cores | {int(r['iterations']):,} sweeps", fontsize=12)
        fig.legend(handles, [h.get_label() for h in handles], loc='upper center',
                   bbox_to_anchor=(.5, .90), frameon=False, fontsize=10, ncol=2)
        fig.text(.5, .025, f"Median of {r['repeats']} repeats; ranges: minimum to maximum. Independent time scales.",
                 ha='center', fontsize=9)
        fig.subplots_adjust(left=.10, right=.98, bottom=.23, top=.72, wspace=.55)
        filename = 'layout_dimensions.png'
        fig.savefig(output/filename, dpi=300); plt.close(fig)
        with (output/'dimension_figure_index.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['figure', *panel_keys])
            writer.writeheader()
            writer.writerows(dict(figure=filename, **{k:records[0][k] for k in panel_keys})
                             for records in overview)
    # Show host staging and CUDA-aware MPI as left/right panels in each
    # dimension's own figure, keeping allocation and numerical work matched.
    path_keys = tuple(k for k in panel_keys if k != 'communication')
    path_panels = defaultdict(list)
    for record in summary:
        if record['mode'] == 'mpi-cuda':
            path_panels[tuple(record[k] for k in path_keys)].append(record)
    path_index = []
    for index, records in enumerate(path_panels.values()):
        if {r['communication'] for r in records} != {'host', 'cuda-aware'}:
            continue  # A single-path study cannot provide a matched path comparison.
        shapes = [{(r['px'], r['py'], r['pz']) for r in records if r['communication'] == path}
                  for path in ('host', 'cuda-aware')]
        if shapes[0] != shapes[1]:
            raise ValueError('Memory-path comparison requires matched process-grid shapes')
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
        handles = None
        for ax, path, title in zip(axes, ('host', 'cuda-aware'),
                                   ('Host staging', 'CUDA-aware MPI')):
            selected = [r for r in records if r['communication'] == path]
            selected.sort(key=lambda r: tuple(int(r['p'+a]) for a in 'xyz'))
            positions = list(range(len(selected)))
            labels = ['x'.join(r['p'+a] for a in 'xyz'[:int(r['dimension'])]) for r in selected]
            series = []
            for overlap, offset, color, label in [
                    (0, -.08, '#176b98', 'No overlap (O=0)'),
                    (1, .08, '#c46b23', 'Interior overlap (O=1)')]:
                times = [r[f'median_seconds_overlap{overlap}'] for r in selected]
                errors = [[t-r[f'min_seconds_overlap{overlap}'] for t, r in zip(times, selected)],
                          [r[f'max_seconds_overlap{overlap}']-t for t, r in zip(times, selected)]]
                series.append(ax.errorbar([x+offset for x in positions], times, yerr=errors,
                              fmt='o' if overlap == 0 else 's', color=color, markersize=5,
                              capsize=3, label=label))
            handles = series
            ax.set(yscale='log', title=title, ylabel='Solve time (s)', xlabel='Process grid',
                   xticks=positions, xticklabels=labels)
            ax.set_xlim(-.4, max(positions)+.4)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=10, labelleft=True)
            ax.grid(axis='y', which='both', alpha=.2)
        # Fix a common range after both paths are drawn: shared-axis log
        # autoscaling can otherwise retain the empty axes' initial limits.
        import matplotlib.ticker as ticker
        low = min(r[f'min_seconds_overlap{o}'] for r in records for o in (0, 1))
        high = max(r[f'max_seconds_overlap{o}'] for r in records for o in (0, 1))
        axes[0].set_ylim(low*.97, high*1.03)
        axes[0].yaxis.set_major_locator(ticker.LogLocator(base=10, subs=tuple(range(1, 10))))
        axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f'{value:g}'))
        axes[0].yaxis.set_minor_locator(ticker.NullLocator())
        r = records[0]
        size = ' x '.join(r['n'+a] for a in 'xyz'[:int(r['dimension'])])
        fig.suptitle(f"{r['dimension']}D: {size}\n{r['gpus_used']} GPU, {r['mpi_ranks']} MPI ranks | "
                     f"{int(r['iterations']):,} sweeps", fontsize=12, y=.99)
        fig.legend(handles, [h.get_label() for h in handles], loc='upper center',
                   bbox_to_anchor=(.5, .84), frameon=False, fontsize=10, ncol=2)
        fig.text(.5, .018, f"Median of {r['repeats']} repeats; ranges: minimum to maximum.",
                 ha='center', fontsize=9)
        fig.subplots_adjust(left=.10, right=.98, bottom=.23, top=.66, wspace=.30)
        filename = f'communication_{index:03d}.png'
        fig.savefig(output/filename, dpi=300)
        plt.close(fig)
        path_index.append(dict(figure=filename, **{k:r[k] for k in path_keys}))
    if path_index:
        with (output/'communication_figure_index.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(path_index[0]))
            writer.writeheader(); writer.writerows(path_index)
    with (output/'figure_index.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(figure_index[0]))
        writer.writeheader()
        writer.writerows(figure_index)
    print(f'Saved {len(summary)} layout summaries and {len(panels)} panels to {output}')


def summarize_transport(rows):
    keys = tuple(k for k in GROUP_KEYS if k != 'communication') + ('overlap',)
    groups = defaultdict(lambda: {'host': {}, 'cuda-aware': {}})
    for row in rows:
        if row['mode'] != 'mpi-cuda' or row['experiment'] != 'layout':
            raise ValueError('Expected CUDA layout measurements')
        transport = row['communication']
        if transport not in ('host', 'cuda-aware'):
            raise ValueError('Unknown transport')
        key = tuple(row[k] for k in keys)
        repeat = int(row['repeat'])
        value = float(row['solve_seconds'])
        if not math.isfinite(value) or value <= 0 or repeat in groups[key][transport]:
            raise ValueError('Invalid time or duplicate repeat')
        groups[key][transport][repeat] = value
    result = []
    for key, data in groups.items():
        host, aware = data['host'], data['cuda-aware']
        if not host or host.keys() != aware.keys():
            raise ValueError('Host and CUDA-aware measurements require matching repeats')
        record = dict(zip(keys, key))
        record.update(repeats=len(host), median_host_seconds=statistics.median(host.values()),
                      median_cuda_aware_seconds=statistics.median(aware.values()),
                      paired_host_over_cuda_aware=statistics.median(host[r]/aware[r] for r in host))
        result.append(record)
    return result


def transport_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = []
    for filename in args.input:
        with Path(filename).open(newline='') as stream:
            rows.extend(csv.DictReader(stream))
    summary = summarize_transport(rows)
    if not summary:
        parser.error('No data')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def audit_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--dim', type=int, choices=[2, 3], required=True)
    p.add_argument('--mode', choices=['mpi-cpu', 'mpi-cuda'], required=True)
    p.add_argument('--process-grids', nargs='+', required=True)
    p.add_argument('--problem-sizes', nargs='+', required=True)
    p.add_argument('--communication', nargs='+', default=['host'], choices=['host', 'cuda-aware'])
    p.add_argument('--iterations', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--job-log', help='Optional CPU job log containing binding smoke and all launch bindings')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if min(a.iterations, a.repeats) < 1:
        p.error('Iterations and repeats must be positive')
    grids = [shape(text, a.dim) for text in a.process_grids]
    sizes = [tuple(map(int, text.split('x'))) for text in a.problem_sizes]
    if any(len(n) != a.dim or min(n) < 1 for n in sizes):
        p.error('Invalid problem size')
    expected = set(itertools.product(sizes, grids, a.communication, (0, 1), range(a.repeats)))
    with Path(a.input).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    found = set()
    numerical = defaultdict(list)
    for row in rows:
        points = tuple(int(row['n'+axis]) for axis in 'xyz'[:a.dim])
        grid = tuple(int(row['p'+axis]) for axis in 'xyz'[:a.dim])
        case = dict(points=points, grid=grid, communication=row['communication'], overlap=int(row['overlap']))
        validate_record(row, case, a.mode, a.dim, a.iterations)
        key = (points, grid, row['communication'], int(row['overlap']), int(row['repeat']))
        if key in found:
            raise ValueError('Duplicate experimental case')
        found.add(key)
        numerical[points].append(row)
    if found != expected:
        raise ValueError(f'Incomplete or extra matrix: missing={expected-found}, extra={found-expected}')
    if len({r['allocation'] for r in rows}) != 1:
        raise ValueError('Audit one allocation/session at a time')
    summary = summarize_layouts(rows)  # Also checks paired repeats and hardware/settings grouping.
    for records in numerical.values():
        for field in ('residual_inf', 'error_inf', 'error_rms'):
            base = float(records[0][field])
            if not all(math.isclose(float(r[field]), base, rel_tol=1e-10, abs_tol=1e-10) for r in records):
                raise ValueError(f'Numerical diagnostics differ across matched layouts: {field}')
    result = dict(status='passed', dimension=a.dim, mode=a.mode, records=len(rows),
                  layout_summaries=len(summary), iterations=a.iterations, repeats=a.repeats,
                  allocation=rows[0]['allocation'], numerical_consistency='passed')
    if a.job_log:
        if a.mode != 'mpi-cpu' or len({math.prod(g) for g in grids}) != 1:
            p.error('Binding-log audit requires a CPU study at fixed rank count')
        ranks = math.prod(grids[0])
        log = Path(a.job_log).read_text()
        completed = any(line.strip() in ('PART3_CPU_LAYOUT_OK', 'PART3_CPU_LAYOUT3D_OK')
                        for line in log.splitlines())
        if 'PMIX ERROR' in log or not completed:
            raise ValueError('Runtime error or missing job completion marker')
        bindings = re.findall(r'Rank (\d+) bound to OS CPU (\d+)', log)
        if len(bindings) != ranks*(len(rows)+1):
            raise ValueError('Missing binding records (one smoke launch plus all timed launches expected)')
        by_rank = defaultdict(set)
        for rank, cpu in bindings:
            by_rank[int(rank)].add(int(cpu))
        if set(by_rank) != set(range(ranks)) or any(len(cpus) != 1 for cpus in by_rank.values()):
            raise ValueError('Rank-to-CPU binding changed within the study')
        if len(set.union(*by_rank.values())) != ranks:
            raise ValueError('Ranks share CPU bindings')
        result.update(binding_records=len(bindings), distinct_cpus=ranks, pmix_errors=0)
    Path(a.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


def bind_main():
    """Bind a local rank to an allocated OS CPU before executing the solver."""
    if sys.argv[1:] == ['--print-cpuset']:
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 16:
            raise RuntimeError(f'Need 16 allocated OS CPUs, got {cpus}')
        print(','.join(map(str, cpus[:16])))
        return
    if len(sys.argv) < 2:
        raise SystemExit('Usage: layout_study.py bind SOLVER [SOLVER_OPTIONS...]')
    rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    cpus = [int(cpu) for cpu in os.environ['POISSON_CPUSET'].split(',')]
    if rank < 0 or rank >= len(cpus):
        raise RuntimeError('Insufficient allocated CPUs for local ranks')
    if cpus[rank] not in os.sched_getaffinity(0):
        raise RuntimeError('Requested CPU is outside the inherited scheduler cpuset')
    os.sched_setaffinity(0, {cpus[rank]})
    print(f'Rank {rank} bound to OS CPU {cpus[rank]}', file=sys.stderr, flush=True)
    solver = str(Path(sys.argv[1]).resolve())
    os.execv(solver, [solver, *sys.argv[2:]])


def main():
    commands = {'benchmark': collect_main, 'plot': plot_main,
                'transport': transport_main, 'audit': audit_main, 'bind': bind_main}
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: layout_study.py {benchmark,plot,transport,audit,bind} [OPTIONS]')
        print('Use a subcommand followed by --help for its options.')
        return
    command = sys.argv.pop(1)
    if command not in commands:
        raise SystemExit(f'Unknown subcommand: {command}')
    commands[command]()


if __name__ == '__main__':
    main()
