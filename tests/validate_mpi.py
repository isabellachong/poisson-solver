#!/usr/bin/env python3
"""Small numerical validation suite. --serial-only needs neither MPI sockets nor GPUs.

MPI mode checks 2x2 (2D) and 2x2x2 (3D) subdomains by default. GPU mode
automatically distributes ranks across visible allocated GPUs, sharing as needed.
"""
import argparse
import csv
import io
import math
from pathlib import Path
import subprocess
import tempfile


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
    rows = list(csv.DictReader(io.StringIO(process.stdout)))
    if len(rows) != 1:
        raise AssertionError(f"Expected one summary row: {process.stdout}")
    return rows[0]


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
    parser.add_argument('--launcher', default='mpiexec')
    parser.add_argument('--launcher-arg', action='append', default=[])
    parser.add_argument('--grid2d', default='2x2')
    parser.add_argument('--grid3d', default='2x2x2')
    parser.add_argument('--communication', choices=['host', 'cuda-aware', 'both'], default='both')
    args = parser.parse_args()
    serial, parallel = str(Path(args.serial).resolve()), str(Path(args.parallel).resolve())
    checked = 0
    with tempfile.TemporaryDirectory(prefix='poisson-validation-') as folder:
        folder = Path(folder)
        for dim in (2, 3):
            n, steps = (9, 7, 5), 9
            expected, residual = reference(dim, n, steps)
            common = ['--dim', dim, '--nx', n[0], '--ny', n[1], '--nz', n[2],
                      '--iterations', steps, '--check-every', 0, '--warmup', 2, '--problem', 'polynomial']
            prefix = folder/f'serial{dim}'
            record = execute([serial, *common, '--solution-prefix', prefix])
            check_solution(prefix, expected)
            assert math.isclose(float(record['residual_inf']), residual, rel_tol=1e-10, abs_tol=1e-9)
            checked += 1
            if args.serial_only:
                continue
            shape = tuple(map(int, (args.grid2d if dim == 2 else args.grid3d).split('x')))
            if len(shape) != dim or min(shape) < 1 or sum(x > 1 for x in shape) < 2:
                parser.error('Distributed grids must split at least two directions')
            ranks = math.prod(shape)
            grid_arguments = [item for axis, value in zip('xyz', shape) for item in ('--p'+axis, value)]
            transports = ['host'] if args.backend == 'cpu' else (['host', 'cuda-aware'] if args.communication == 'both' else [args.communication])
            for transport in transports:
                for overlap in (0, 1):
                    prefix = folder/f'{args.backend}{dim}_{transport}_{overlap}'
                    command = [args.launcher, *args.launcher_arg, '-n', ranks, parallel, *common, *grid_arguments,
                               '--overlap', overlap, '--communication', transport, '--solution-prefix', prefix]
                    record = execute(command)
                    assert record['mode'] == ('mpi-cuda' if args.backend == 'cuda' else 'mpi-cpu')
                    assert int(record['mpi_ranks']) == ranks
                    if args.backend == 'cuda':
                        assert int(record['gpus_used']) >= 1
                        assert 1 <= int(record['max_ranks_per_gpu']) <= ranks
                    check_solution(prefix, expected)
                    assert math.isclose(float(record['residual_inf']), residual, rel_tol=1e-10, abs_tol=1e-9)
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
