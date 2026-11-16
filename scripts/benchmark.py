#!/usr/bin/env python3
"""Collect real solver measurements inside an existing allocation; no job submission.

Examples: --mode mpi-cpu --kind strong --process-grids 1x1 2x2 4x4 8x8 16x16
          --mode mpi-cuda --kind sharing --process-grids 1x1 2x2 4x4 (one visible GPU)
GPU weak scaling requires a constant measured rank/GPU ratio; sharing studies
require a constant measured GPU count. Actual GPU counts always come from the solver.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.solver_output import parse_summary


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
    p.add_argument('--problem', choices=['sine', 'mixed', 'polynomial', 'zero'], default='mixed')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--label', default='', help='Experiment environment label, e.g. vanda-model-allocation')
    p.add_argument('--timeout', type=float, default=3600, help='Maximum seconds per solver invocation')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--overlap', type=int, choices=[0, 1], nargs='+', default=[0, 1])
    p.add_argument('--launcher', default='mpiexec')
    p.add_argument('--launcher-arg', action='append', default=[])
    p.add_argument('--communication', choices=['host', 'cuda-aware'], default='cuda-aware')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.repeats < 1 or a.iterations < 1 or min(a.nx, a.ny, a.nz, *a.sizes) < 1:
        p.error('Repeats, dimensions, and iteration counts must be positive')
    if a.warmup < 0 or not math.isfinite(a.timeout) or a.timeout <= 0:
        p.error('Warmup must be nonnegative and timeout must be finite and positive')
    if (a.kind == 'serial') != (a.mode == 'serial'):
        p.error('Serial mode requires --kind serial, and vice versa')
    if a.kind == 'sharing' and a.mode != 'mpi-cuda':
        p.error('A sharing study requires mpi-cuda mode')
    executable = str(Path(a.exe).resolve())
    if not Path(executable).is_file():
        p.error(f'Executable does not exist: {executable}')
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
    metadata = output.with_suffix(output.suffix + '.metadata.json')
    log = output.with_suffix(output.suffix + '.runs.jsonl')
    if any(path.exists() for path in (output, metadata, log)):
        p.error('Output or provenance files already exist; choose a new output filename')
    metadata.write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
        arguments=vars(a), platform=platform.platform(),
        host=platform.node(), python=sys.version, executable=executable,
        executable_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
        note='Host describes the launcher; record compute-node CPU/GPU, allocation, compiler, MPI, CUDA and MPS separately.'), indent=2)+'\n')
    writer = None
    gpu_reference = None
    with output.open('x', newline='') as stream, log.open('x') as log_stream:
        for grid, points in cases:
            ranks = math.prod(grid)
            for repeat in range(a.repeats):
                schedules = [0] if a.mode == 'serial' else list(a.overlap)
                if repeat % 2:
                    schedules.reverse()
                for overlap in schedules:
                    command = [] if a.mode == 'serial' else [a.launcher, *a.launcher_arg, '-n', str(ranks)]
                    command += [executable, '--dim', str(a.dim), '--iterations', str(a.iterations),
                                '--check-every', '0', '--overlap', str(overlap), '--warmup', str(a.warmup),
                                '--problem', a.problem,
                                '--communication', a.communication]
                    for axis, n, count in zip('xyz', points, grid):
                        command += ['--n'+axis, str(n), '--p'+axis, str(count)]
                    print(f'{a.kind}: grid={grid}, points={points}, repeat={repeat}, overlap={overlap}', file=sys.stderr)
                    log_stream.write(json.dumps(dict(event='start', command=command, repeat=repeat))+'\n')
                    log_stream.flush()
                    run = subprocess.run(command, text=True, capture_output=True, timeout=a.timeout)
                    log_stream.write(json.dumps(dict(event='finish', returncode=run.returncode,
                                                    stdout=run.stdout, stderr=run.stderr))+'\n')
                    log_stream.flush()
                    print(run.stderr, end='', file=sys.stderr)
                    if run.returncode:
                        raise RuntimeError(f'Solver failed ({run.returncode}): {run.stdout}')
                    row = parse_summary(run.stdout)
                    if row['mode'] != a.mode:
                        raise RuntimeError('Unexpected solver mode or output')
                    expected = dict(dimension=a.dim, mpi_ranks=ranks, iterations=a.iterations,
                                    check_every=0, warmup=a.warmup)
                    expected.update({'n'+axis: n for axis, n in zip('xyz', points)})
                    expected.update({'p'+axis: n for axis, n in zip('xyz', grid)})
                    if any(int(row[key]) != value for key, value in expected.items()) or row['problem'] != a.problem:
                        raise RuntimeError('Solver output does not match the requested experiment')
                    if not math.isfinite(float(row['solve_seconds'])) or float(row['solve_seconds']) <= 0:
                        raise RuntimeError('Solver returned invalid timing')
                    if a.mode == 'mpi-cuda' and a.kind in ('weak', 'sharing'):
                        gpus = int(row['gpus_used'])
                        measured = ranks/gpus if a.kind == 'weak' else gpus
                        if gpu_reference is None:
                            gpu_reference = measured
                        elif measured != gpu_reference:
                            raise RuntimeError('GPU resources changed inconsistently: weak scaling needs constant ranks/GPU; sharing studies need constant GPU count. Earlier rows remain in the CSV.')
                    record = {'experiment': a.kind, 'repeat': repeat, 'label': a.label, **row}
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(record))
                        writer.writeheader()
                    writer.writerow(record)
                    stream.flush()
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
