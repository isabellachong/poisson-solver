#!/usr/bin/env python3
"""Measure spatial accuracy using converged sine solves, separate from timings."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from validate_mpi import check_solution, execute


def sine_reference(dimension, n, iterations):
    """Independent finite-step solution from the discrete sine eigenvalue.

    No stencil update loop is reused. This checks the actual owned values,
    including the iteration count and the reset after untimed warmup.
    """
    sines = [math.sin(math.pi*i/(n+1)) for i in range(1, n+1)]
    continuous = dimension*math.pi**2
    discrete = 4*dimension*(n+1)**2*math.sin(math.pi/(2*(n+1)))**2
    rho = math.cos(math.pi/(n+1))
    amplitude = continuous/discrete*(1-rho**iterations)
    expected = {}
    for k in (range(1, n+1) if dimension == 3 else [0]):
        for j in range(1, n+1):
            for i in range(1, n+1):
                basis = sines[i-1]*sines[j-1]*(sines[k-1] if dimension == 3 else 1)
                expected[(i, j, k)] = amplitude*basis
    error_inf = abs(amplitude-1)*max(sines)**dimension
    error_rms = abs(amplitude-1)*(sum(s*s for s in sines)/n)**(dimension/2)
    return expected, error_inf, error_rms


def plot_refinement(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, dimension, color, marker in zip(axes, (2, 3), ('#176b98', '#c46b23'), ('o', 's')):
        selected = [row for row in rows if row['dimension'] == dimension]
        h = [row['h'] for row in selected]
        error = [row['error_inf'] for row in selected]
        ax.plot(h, error, marker=marker, color=color, label='Measured maximum error')
        ax.plot(h, [error[0]*(value/h[0])**2 for value in h], '--',
                color='#666666', label='Second-order reference')
        ax.set(xlabel='Grid spacing h', ylabel='Maximum solution error',
               title=f'{dimension}D sine problem')
        ax.grid(True, which='both', alpha=.18)
        ax.legend(fontsize=8)
    for ax in axes:
        ax.set(xscale='log', yscale='log')
    fig.suptitle('Jacobi accuracy: sine solutions converged to residual tolerance', fontsize=12)
    fig.tight_layout()
    fig.savefig(output/'refinement.png', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', default='build-cpu/executables/poisson_serial')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--points', type=int, nargs='+', default=[7, 15, 31])
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if len(args.points) < 2 or args.points != sorted(set(args.points)) or min(args.points) < 3:
        parser.error('Use at least two strictly increasing grid sizes, each at least three')
    executable = str(Path(args.exe).resolve())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows, runs = [], []
    for dimension in (2, 3):
        previous = None
        for n in args.points:
            prefix = output/f'solution-{dimension}d-n{n}'
            command = [executable, '--dim', dimension, '--nx', n, '--ny', n, '--nz', n,
                       '--problem', 'sine', '--iterations', 50000, '--check-every', 10,
                       '--warmup', 2, '--rtol', '1e-11', '--atol', '1e-12',
                       '--solution-prefix', prefix]
            summary = execute(command)
            if summary['mode'] != 'serial' or int(summary['mpi_ranks']) != 1:
                raise AssertionError('The spatial-accuracy study requires the serial executable')
            residual, threshold = float(summary['residual_inf']), float(summary['threshold'])
            if (summary['converged'] != '1' or not math.isfinite(residual)
                    or not 0 <= residual <= threshold):
                raise AssertionError('Refinement solve did not converge to the required tolerance')
            iterations = int(summary['iterations'])
            expected, reference_inf, reference_rms = sine_reference(dimension, n, iterations)
            check_solution(prefix, expected)
            error, rms = float(summary['error_inf']), float(summary['error_rms'])
            for actual, reference in [(error, reference_inf), (rms, reference_rms)]:
                if not math.isclose(actual, reference, rel_tol=1e-9, abs_tol=1e-12):
                    raise AssertionError('Reported error disagrees with the independent sine-mode reference')
            if not math.isfinite(error) or error <= 0:
                raise AssertionError('Expected a finite positive spatial error')
            h = 1/(n+1)
            order = math.log(previous['error_inf']/error)/math.log(previous['h']/h) if previous else ''
            if order != '' and not 1.9 < order < 2.1:
                raise AssertionError(f'Expected second-order accuracy, observed {order}')
            row = dict(dimension=dimension, n=n, h=h, error_inf=error, error_rms=rms,
                       observed_order=order, iterations=iterations, residual_inf=residual,
                       threshold=threshold, solve_seconds=float(summary['solve_seconds']),
                       independent_reference='passed')
            rows.append(row)
            runs.append(dict(command=list(map(str, command)), summary=summary))
            previous = row
            print(f'{dimension}D N={n}: h={h:.8g}, iterations={iterations}, '
                  f'max_error={error:.10g}, order={order}', flush=True)
    with (output/'refinement.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(executable=executable,
                    executable_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                    host=platform.node(), platform=platform.platform(), python=sys.version,
                    pbs_job_id=os.environ.get('PBS_JOBID'),
                    note='Converged spatial-accuracy study; separate from fixed-iteration performance experiments.',
                    runs=runs)
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.plot:
        plot_refinement(rows, output)
    print(f'Passed {len(rows)} converged refinement solves and independent owned-value checks; saved {output}')


if __name__ == '__main__':
    main()
