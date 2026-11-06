#!/usr/bin/env python3
"""Collect real solver measurements inside an existing allocation; no job submission.

Examples: --mode mpi-cpu --kind strong --process-grids 1x1 2x2 4x4 8x8 16x16
          --mode mpi-cuda --kind sharing --process-grids 1x1 2x2 4x4 (one visible GPU)
GPU weak scaling requires a constant measured rank/GPU ratio; sharing studies
require a constant measured GPU count. Actual GPU counts always come from the solver.
"""
import argparse
import csv
import io
import math
from pathlib import Path
import subprocess
import sys


def shape(text, dimension):
    values = tuple(map(int, text.split('x')))
    if len(values) != dimension or min(values) < 1:
        raise ValueError(f'Expected {dimension} positive process dimensions: {text}')
    if math.prod(values) > 1 and sum(value > 1 for value in values) < 2:
        raise ValueError('Process grids must split at least two axes')
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', required=True)
    p.add_argument('--mode', required=True, choices=['serial', 'mpi-cpu', 'mpi-cuda'])
    p.add_argument('--kind', required=True, choices=['serial', 'strong', 'weak', 'sharing'])
    p.add_argument('--dim', type=int, choices=[2, 3], default=2)
    p.add_argument('--process-grids', nargs='+')
    p.add_argument('--sizes', type=int, nargs='+', default=[32, 64, 128])
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--ny', type=int, default=128)
    p.add_argument('--nz', type=int, default=128)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--overlap', type=int, choices=[0, 1], nargs='+', default=[0, 1])
    p.add_argument('--launcher', default='mpiexec')
    p.add_argument('--launcher-arg', action='append', default=[])
    p.add_argument('--communication', choices=['host', 'cuda-aware'], default='cuda-aware')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.repeats < 1 or a.iterations < 1 or min(a.nx, a.ny, a.nz, *a.sizes) < 1:
        p.error('Repeats, dimensions, and iteration counts must be positive')
    if (a.kind == 'serial') != (a.mode == 'serial'):
        p.error('Serial mode requires --kind serial, and vice versa')
    if a.kind == 'sharing' and a.mode != 'mpi-cuda':
        p.error('A sharing study requires mpi-cuda mode')
    executable = str(Path(a.exe).resolve())
    if a.mode == 'serial':
        cases = [((1,)*a.dim, (n,)*a.dim) for n in a.sizes]
    else:
        if not a.process_grids:
            p.error('MPI modes require explicit --process-grids')
        try:
            grids = [shape(text, a.dim) for text in a.process_grids]
        except ValueError as e:
            p.error(str(e))
        base = (a.nx, a.ny, a.nz)[:a.dim]
        cases = [(grid, tuple(n*q for n, q in zip(base, grid)) if a.kind == 'weak' else base) for grid in grids]
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    gpu_reference = None
    with output.open('x', newline='') as stream:
        for grid, points in cases:
            ranks = math.prod(grid)
            for repeat in range(a.repeats):
                schedules = [0] if a.mode == 'serial' else list(a.overlap)
                if repeat % 2:
                    schedules.reverse()
                for overlap in schedules:
                    command = [] if a.mode == 'serial' else [a.launcher, *a.launcher_arg, '-n', str(ranks)]
                    command += [executable, '--dim', str(a.dim), '--iterations', str(a.iterations),
                                '--check-every', '0', '--overlap', str(overlap), '--warmup', '3',
                                '--communication', a.communication]
                    for axis, n, count in zip('xyz', points, grid):
                        command += ['--n'+axis, str(n), '--p'+axis, str(count)]
                    print(f'{a.kind}: grid={grid}, points={points}, repeat={repeat}, overlap={overlap}', file=sys.stderr)
                    run = subprocess.run(command, text=True, capture_output=True)
                    print(run.stderr, end='', file=sys.stderr)
                    if run.returncode:
                        raise RuntimeError(f'Solver failed ({run.returncode}): {run.stdout}')
                    rows = list(csv.DictReader(io.StringIO(run.stdout)))
                    if len(rows) != 1 or rows[0]['mode'] != a.mode:
                        raise RuntimeError('Unexpected solver mode or output')
                    row = rows[0]
                    if a.mode == 'mpi-cuda' and a.kind in ('weak', 'sharing'):
                        gpus = int(row['gpus_used'])
                        measured = ranks/gpus if a.kind == 'weak' else gpus
                        if gpu_reference is None:
                            gpu_reference = measured
                        elif measured != gpu_reference:
                            raise RuntimeError('GPU resources changed inconsistently: weak scaling needs constant ranks/GPU; sharing studies need constant GPU count. Earlier rows remain in the CSV.')
                    record = {'experiment': a.kind, 'repeat': repeat, **row}
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(record))
                        writer.writeheader()
                    writer.writerow(record)
                    stream.flush()
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
