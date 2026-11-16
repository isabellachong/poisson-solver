#!/usr/bin/env python3
"""Small numerical validation suite. --serial-only needs neither MPI sockets nor GPUs.

MPI mode checks 2x2 (2D) and 2x2x2 (3D) subdomains by default. GPU mode
automatically distributes ranks across visible allocated GPUs, sharing as needed.
"""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.solver_output import parse_summary


def exact(dim, xyz):
    x, y, z = xyz
    return x + 2*y + (3*z if dim == 3 else 0) + x*(1-x)*y*(1-y)*(z*(1-z) if dim == 3 else 1)


def reference(dim, n, steps):
    nx, ny, nz = n
    nz = nz if dim == 3 else 1
    sx, sy = nx+2, ny+2
    size = sx*sy*(nz+2 if dim == 3 else 1)
    values, forcing = [0.0]*size, [0.0]*size
    scale = [(n[a]+1)**2 for a in range(dim)]
    def index(i, j, k):
        return i + sx*(j+sy*k)
    owned = [(i, j, k) for k in (range(1, nz+1) if dim == 3 else [0])
             for j in range(1, ny+1) for i in range(1, nx+1)]
    for k in (range(nz+2) if dim == 3 else [0]):
        for j in range(ny+2):
            for i in range(nx+2):
                xyz = [i/(nx+1), j/(ny+1), k/(nz+1) if dim == 3 else 0]
                if i in (0, nx+1) or j in (0, ny+1) or (dim == 3 and k in (0, nz+1)):
                    values[index(i, j, k)] = exact(dim, xyz)
    for i, j, k in owned:
        x, y, z = i/(nx+1), j/(ny+1), k/(nz+1) if dim == 3 else 0
        a, b, c = x*(1-x), y*(1-y), z*(1-z)
        forcing[index(i, j, k)] = 2*(a+b) if dim == 2 else 2*(a*b+a*c+b*c)
    strides = [1, sx, sx*sy]
    for _ in range(steps):
        next_values = values.copy()
        for i, j, k in owned:
            q = index(i, j, k)
            next_values[q] = (forcing[q] + sum(scale[a]*(values[q-strides[a]]+values[q+strides[a]])
                                              for a in range(dim))) / (2*sum(scale))
        values = next_values
    residual = 0.0
    for i, j, k in owned:
        q = index(i, j, k)
        ahu = sum(scale[a]*(2*values[q]-values[q-strides[a]]-values[q+strides[a]]) for a in range(dim))
        residual = max(residual, abs(forcing[q]-ahu))
    return {(i, j, k): values[index(i, j, k)] for i, j, k in owned}, residual


def execute(command):
    process = subprocess.run(list(map(str, command)), text=True, capture_output=True, timeout=180)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}): {command}\n{process.stdout}\n{process.stderr}")
    print(process.stderr, end='', file=sys.stderr)
    return parse_summary(process.stdout)


def check_solution(prefix, expected):
    found = {}
    for path in prefix.parent.glob(prefix.name+'.rank*.csv'):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = tuple(int(row['global_'+a]) for a in 'ijk')
                if key in found:
                    raise AssertionError(f'Duplicate ownership at {key}')
                found[key] = float(row['u'])
    if found.keys() != expected.keys():
        raise AssertionError('Missing or extra owned points')
    for key, target in expected.items():
        if not math.isclose(found[key], target, rel_tol=1e-11, abs_tol=1e-12):
            raise AssertionError(f'Numerical mismatch at {key}: {found[key]} versus {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', default='build-cpu/executables/poisson_serial')
    parser.add_argument('--parallel', default='build-cpu/executables/poisson_mpi')
    parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--serial-only', action='store_true')
    parser.add_argument('--work-dir', type=Path,
                        help='Shared directory for per-rank solutions; defaults to the current working directory')
    parser.add_argument('--suite', choices=['basic', 'part3'], default='basic',
                        help='part3 adds uneven, rectangular and thin Cartesian subdomains (up to 8 ranks)')
    parser.add_argument('--launcher', default='mpiexec')
    parser.add_argument('--launcher-arg', action='append', default=[])
    parser.add_argument('--grid2d', default='2x2')
    parser.add_argument('--grid3d', default='2x2x2')
    parser.add_argument('--points', type=int, nargs=3, default=(9, 7, 5), metavar=('NX', 'NY', 'NZ'))
    parser.add_argument('--communication', choices=['host', 'cuda-aware', 'both'], default='both')
    args = parser.parse_args()
    if min(args.points) < 1:
        parser.error('Reference point counts must be positive')
    serial, parallel = str(Path(args.serial).resolve()), str(Path(args.parallel).resolve())
    checked = 0
    work_dir = args.work_dir.resolve() if args.work_dir else Path.cwd()
    if work_dir:
        work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='poisson-validation-', dir=work_dir) as folder:
        folder = Path(folder)
        cases = [(2, tuple(args.points), args.grid2d), (3, tuple(args.points), args.grid3d)]
        if args.suite == 'part3':
            cases += [(2, (11, 7, 1), '2x3'), (2, (7, 11, 1), '3x2'),
                      (2, (3, 3, 1), '2x2'), (2, (2, 2, 1), '2x2'),
                      (3, (9, 7, 5), '2x2x1'), (3, (7, 5, 3), '1x2x2'),
                      (3, (3, 3, 3), '2x2x2'), (3, (2, 2, 2), '2x2x2')]
        for case_id, (dim, n, grid_text) in enumerate(cases):
            steps = 9
            expected, residual = reference(dim, n, steps)
            common = ['--dim', dim, '--nx', n[0], '--ny', n[1], '--nz', n[2],
                      '--iterations', steps, '--check-every', 0, '--warmup', 2, '--problem', 'polynomial']
            prefix = folder/f'case{case_id}_serial{dim}'
            record = execute([serial, *common, '--solution-prefix', prefix])
            check_solution(prefix, expected)
            assert math.isclose(float(record['residual_inf']), residual, rel_tol=1e-10, abs_tol=1e-9)
            checked += 1
            if args.serial_only:
                continue
            shape = tuple(map(int, grid_text.split('x')))
            if len(shape) != dim or min(shape) < 1 or (math.prod(shape) > 1 and sum(x > 1 for x in shape) < 2):
                parser.error('Use one rank or a distributed grid that splits at least two directions')
            ranks = math.prod(shape)
            if any(p > count for p, count in zip(shape, n)):
                parser.error('Process grid would create an empty subdomain')
            grid_arguments = [item for axis, value in zip('xyz', shape) for item in ('--p'+axis, value)]
            transports = ['host'] if args.backend == 'cpu' else (['host', 'cuda-aware'] if args.communication == 'both' else [args.communication])
            for transport in transports:
                for overlap in (0, 1):
                    prefix = folder/f'case{case_id}_{args.backend}{dim}_{transport}_{overlap}'
                    command = [args.launcher, *args.launcher_arg, '-n', ranks, parallel, *common, *grid_arguments,
                               '--overlap', overlap, '--communication', transport, '--solution-prefix', prefix]
                    record = execute(command)
                    assert record['mode'] == ('mpi-cuda' if args.backend == 'cuda' else 'mpi-cpu')
                    assert int(record['mpi_ranks']) == ranks
                    actual_shape = tuple(int(record['p'+axis]) for axis in 'xyz'[:dim])
                    assert actual_shape == shape, (actual_shape, shape)
                    if args.backend == 'cuda':
                        assert int(record['gpus_used']) >= 1
                        assert 1 <= int(record['max_ranks_per_gpu']) <= ranks
                    check_solution(prefix, expected)
                    assert math.isclose(float(record['residual_inf']), residual, rel_tol=1e-10, abs_tol=1e-9)
                    checked += 1
            print(f'Passed case {case_id}: {dim}D points={n[:dim]} grid={shape}')
        if args.suite == 'part3' and not args.serial_only:
            # Exercise residual refresh and MPI_Allreduce inside the iteration loop,
            # not just the final fixed-iteration diagnostic.
            for dim in (2, 3):
                shape = (2,)*dim
                grid_arguments = [item for a, value in zip('xyz', shape) for item in ('--p'+a, value)]
                transports = ['host'] if args.backend == 'cpu' else (['host', 'cuda-aware'] if args.communication == 'both' else [args.communication])
                target = {(i, j, k): exact(dim, (i/4, j/4, k/4 if dim == 3 else 0))
                          for k in (range(1, 4) if dim == 3 else [0])
                          for j in range(1, 4) for i in range(1, 4)}
                for transport in transports:
                    for overlap in (0, 1):
                        prefix = folder/f'converged{dim}_{transport}_{overlap}'
                        command = [args.launcher, *args.launcher_arg, '-n', math.prod(shape), parallel,
                                   '--dim', dim, '--nx', 3, '--ny', 3, '--nz', 3,
                                   *grid_arguments, '--problem', 'polynomial', '--iterations', 10000,
                                   '--check-every', 3, '--rtol', 1e-12, '--atol', 1e-12,
                                   '--overlap', overlap, '--communication', transport,
                                   '--solution-prefix', prefix]
                        record = execute(command)
                        assert record['converged'] == '1'
                        assert 0 < int(record['iterations']) < 10000
                        assert float(record['residual_inf']) <= float(record['threshold'])
                        check_solution(prefix, target)
                        checked += 1
        # Initial convergence and nonconvergence are separate, purposeful checks.
        zero = execute([serial, '--dim', 3, '--nx', 3, '--ny', 3, '--nz', 3,
                        '--problem', 'zero', '--check-every', 1, '--iterations', 10, '--warmup', 1])
        assert zero['converged'] == '1' and zero['iterations'] == '0'
        failed = subprocess.run([serial, '--dim', '2', '--nx', '7', '--ny', '5', '--iterations', '1',
                                 '--check-every', '2', '--warmup', '0'], text=True, capture_output=True, timeout=30)
        assert failed.returncode == 2, failed.stderr
        checked += 2
    print(f'Passed {checked} executable checks (independent numerical reference, owned values, residuals, and stopping behaviour)')


if __name__ == '__main__':
    main()
