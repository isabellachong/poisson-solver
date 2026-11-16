"""18 report regression checks; synthetic fixtures never become performance evidence.

Build the serial executable first. Run with unittest discovery from the root.
"""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from layout_study import build_plan, summarize_layouts as summarize, validate_record, summarize_transport
from solver_output import parse_summary


ROOT = Path(__file__).resolve().parents[1]

class AnalysisScripts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.exe = Path(os.environ.get('POISSON_SERIAL', ROOT / 'build-cpu/executables/poisson_serial')).resolve()
        if not self.exe.is_file():
            self.skipTest('Build serial executable first or set POISSON_SERIAL')
        self.csv = self.folder / 'serial.csv'
        self.command = [sys.executable, str(ROOT / 'scripts/benchmark.py'), '--exe', str(self.exe),
                        '--mode', 'serial', '--kind', 'serial', '--sizes', '7', '--iterations', '2',
                        '--repeats', '2', '--output', str(self.csv), '--label', 'test']
        self.run_command(self.command)
        with self.csv.open() as stream:
            self.serial = list(csv.DictReader(stream))[0]

    def run_command(self, command, success=True):
        env = dict(os.environ, MPLCONFIGDIR=str(self.folder / 'mpl'))
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def plot(self, rows, success=True, serial=False):
        path = self.folder / 'parallel.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        command = [sys.executable, str(ROOT / 'scripts/plot_scaling.py'), '--input', str(path),
                   '--output-dir', str(self.folder / 'figures')]
        if serial:
            command += ['--serial-reference', str(self.csv)]
        return self.run_command(command, success)

    def records(self):
        return [dict(self.serial, experiment='strong', mode='mpi-cpu', mpi_ranks=str(n),
                     communication='host', solve_seconds=str(t), px=str(p), py=str(p))
                for n, p, t in ((1, 1, 4), (4, 2, 2))]

    def test_provenance_and_no_overwrite(self):
        metadata = json.loads(self.csv.with_suffix('.csv.metadata.json').read_text())
        self.assertEqual(len(metadata['executable_sha256']), 64)
        log = [json.loads(line) for line in self.csv.with_suffix('.csv.runs.jsonl').read_text().splitlines()]
        self.assertEqual([r['event'] for r in log], ['start', 'finish', 'start', 'finish'])
        before = self.csv.read_bytes()
        self.run_command(self.command, success=False)
        self.assertEqual(self.csv.read_bytes(), before)

    def test_reject_equal_size_different_grid(self):
        rows = self.records()
        rows[1].update(nx='49', ny='1')  # Still 49 unknowns, different geometry.
        self.assertIn('grid shape', self.plot(rows, success=False).stderr)

    def test_reject_nonfinite_sample(self):
        rows = self.records()
        rows.append(dict(rows[1], solve_seconds='nan'))
        self.assertIn('finite and positive', self.plot(rows, success=False).stderr)

    def test_speedup_and_serial_reference(self):
        self.plot(self.records(), serial=True)
        with (self.folder / 'figures/summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(float(rows[1]['relative_speedup']), 2)
        self.assertEqual(float(rows[1]['efficiency']), .5)
        self.assertAlmostEqual(float(rows[1]['speedup_vs_serial']), float(rows[1]['serial_seconds']) / 2)
        self.assertEqual(len(list((self.folder / 'figures').glob('*.png'))), 4)

    def test_reject_unmatched_serial_reference(self):
        rows = self.records()
        for row in rows:
            row['label'] = 'different-machine'
        self.assertIn('No serial reference matches', self.plot(rows, success=False, serial=True).stderr)

    def test_separate_environment_and_warmup_series(self):
        for field, value in [('label', 'another-session'), ('warmup', '30')]:
            first = self.records()[1]
            second = dict(first, **{field: value}, solve_seconds='9')
            self.plot([first, second])
            with (self.folder / 'figures/summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual({float(r['median_seconds']) for r in rows}, {2, 9})

    def test_reject_pooling_process_grid_shapes(self):
        first = dict(self.records()[1], mpi_ranks='16', px='2', py='8')
        second = dict(first, px='4', py='4', solve_seconds='9')
        self.assertIn('Different process grids', self.plot([first, second], success=False).stderr)

def options(**changes):
    values = dict(dim=2, process_grids=['2x8', '4x4'], problem_sizes=['101x97'],
                  repeats=2, iterations=9, timeout=30, mode='mpi-cpu',
                  communication=['host'], launcher='mpiexec', launcher_arg=[], exe='poisson_mpi')
    values.update(changes)
    return SimpleNamespace(**values)

def fixture(grid='2x8', overlap=0, repeat=0, seconds=2):
    px, py = grid.split('x')
    return dict(experiment='layout', allocation='UNIT_TEST_FIXTURE', mode='mpi-cpu',
                dimension='2', nx='101', ny='97', nz='1', unknowns=str(101*97),
                mpi_ranks='16', px=px, py=py, pz='1', gpus_used='0', max_ranks_per_gpu='0',
                device_name='CPU', communication='host', problem='mixed', iterations='9',
                check_every='0', warmup='3', block_size='0', overlap=str(overlap),
                repeat=str(repeat), solve_seconds=str(seconds), residual_inf='1', error_inf='1', error_rms='1')

class ExperimentIntegrity(unittest.TestCase):
    def test_transport_pairing_and_hardware_separation(self):
        rows = []
        for transport, values in [('host', [2, 100, 100]), ('cuda-aware', [1, 50, 1])]:
            for repeat, seconds in enumerate(values):
                rows.append(dict(fixture(repeat=repeat, seconds=seconds), mode='mpi-cuda',
                                 gpus_used='1', max_ranks_per_gpu='16', communication=transport))
        result = summarize_transport(rows)
        self.assertEqual(result[0]['paired_host_over_cuda_aware'], 2)
        self.assertNotEqual(result[0]['paired_host_over_cuda_aware'],
                            result[0]['median_host_seconds']/result[0]['median_cuda_aware_seconds'])
        with self.assertRaises(ValueError):
            summarize_transport(rows[:-1])
        with self.assertRaises(ValueError):
            summarize_transport(rows + [rows[0]])
        changed = [dict(row) for row in rows]
        changed[-1]['gpus_used'] = '2'
        with self.assertRaises(ValueError):
            summarize_transport(changed)

    def test_plan_matches_work_and_reverses_order(self):
        plan = build_plan(options())
        self.assertEqual(len(plan), 8)
        self.assertEqual([c['command'] for c in plan[:4]],
                         [c['command'] for c in reversed(plan[4:])])
        self.assertTrue(all(c['points'] == (101, 97) for c in plan))

    def test_reject_invalid_comparisons(self):
        for args in [options(process_grids=['2x2', '4x4']),
                     options(process_grids=['1x16']),
                     options(problem_sizes=['3x3']),
                     options(mode='mpi-cpu', communication=['cuda-aware'])]:
            with self.assertRaises(ValueError):
                build_plan(args)

    def test_result_must_match_command_and_be_finite(self):
        case = build_plan(options())[0]
        row = fixture()
        validate_record(row, case, 'mpi-cpu', 2, 9)
        for key, bad in [('px', '4'), ('iterations', '8'), ('solve_seconds', 'nan'),
                         ('residual_inf', 'inf'), ('unknowns', '9')]:
            changed = dict(row, **{key: bad})
            with self.assertRaises(ValueError):
                validate_record(changed, case, 'mpi-cpu', 2, 9)

    def test_summary_keeps_shapes_and_paired_ratios(self):
        rows = [fixture(g, o, r, seconds=(2 if o == 0 else 1)*(r+1))
                for g in ('2x8', '4x4') for o in (0, 1) for r in (0, 1)]
        summary = summarize(rows)
        self.assertEqual(len(summary), 2)
        self.assertTrue(all(r['paired_overlap_speedup'] == 2 for r in summary))
        self.assertEqual({r['px'] for r in summary}, {'2', '4'})

    def test_summary_rejects_unpaired_and_duplicate_repeats(self):
        for rows in [[fixture()], [fixture(), fixture(), fixture(overlap=1)]]:
            with self.assertRaises(ValueError):
                summarize(rows)
        a, b = fixture(), fixture(overlap=1)
        b['gpus_used'] = '1'
        with self.assertRaises(ValueError):
            summarize([a, b])

    def test_plot_cli_keeps_shapes_and_renders(self):
        # Deliberately synthetic input, isolated in a temporary test directory.
        with tempfile.TemporaryDirectory(prefix='part3-fixture-') as folder:
            folder = Path(folder)
            rows = [fixture(g, o) for g in ('2x8', '4x4') for o in (0, 1)]
            source = folder/'UNIT_TEST_FIXTURE.csv'
            with source.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            script = Path(__file__).resolve().parents[1]/'scripts'/'layout_study.py'
            run = subprocess.run([sys.executable, str(script), 'plot', '--input', str(source),
                                  '--output-dir', str(folder/'figures')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            with (folder/'figures'/'layout_summary.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            self.assertEqual(len(list((folder/'figures').glob('*.png'))), 1)

class SolverOutput(unittest.TestCase):
    # Parser fixtures only; these numbers are never performance evidence.
    header = 'mode,dimension,nx,ny,nz,solve_seconds,device_name'
    record = 'serial,2,7,5,1,0.1,CPU'

    def test_diagnostics_are_preserved_and_quoted_fields_parse(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            row = parse_summary('launcher warning\n'+self.header+'\n'+
                                'mpi-cuda,2,7,5,1,0.1,"GPU, fixture"\ntrailer\n')
        self.assertEqual(row['device_name'], 'GPU, fixture')
        self.assertIn('launcher warning', stderr.getvalue())
        self.assertIn('trailer', stderr.getvalue())

    def test_duplicate_missing_or_reordered_records_are_rejected(self):
        for output in ['', self.header, self.record+'\n'+self.header,
                       self.header+'\n'+self.header+'\n'+self.record,
                       self.header+'\n'+self.record+'\n'+self.record]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_summary(output)

    def test_wrong_column_count_or_duplicate_columns_are_rejected(self):
        for output in [self.header+'\nserial,2,7,5,1',
                       self.header+',nx\n'+self.record+',7']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_summary(output)

    def test_malformed_csv_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_summary(self.header+'\nserial,2,7,5,1,0.1,"unterminated')

if __name__ == '__main__':
    unittest.main()
