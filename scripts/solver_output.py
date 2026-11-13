"""Read a solver CSV summary while retaining launcher diagnostics on stderr."""
import csv
import sys


def parse_summary(output):
    lines = output.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith('mode,dimension,nx,ny,nz,')]
    records = [i for i, line in enumerate(lines) if line.startswith(('serial,', 'mpi-cpu,', 'mpi-cuda,'))]
    if len(headers) != 1 or len(records) != 1 or headers[0] >= records[0]:
        raise ValueError(f'Expected one solver CSV header and one summary row:\n{output}')
    header_index, record_index = headers[0], records[0]
    try:
        header, values = list(csv.reader([lines[header_index], lines[record_index]], strict=True))
    except csv.Error as error:
        raise ValueError(f'Malformed solver CSV summary:\n{output}') from error
    if len(header) != len(values) or len(header) != len(set(header)):
        raise ValueError(f'Solver CSV columns do not match:\n{output}')
    for index, line in enumerate(lines):
        if index not in (header_index, record_index) and line.strip():
            print(line, file=sys.stderr)
    return dict(zip(header, values))

