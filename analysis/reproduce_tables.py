"""Reproduce existing derived tables; no predictions or scientific settings change."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError('Use an output directory outside the immutable release.')
    out.mkdir(parents=True, exist_ok=False)
    primary = out / 'primary_r18'
    primary.mkdir()
    shutil.copy2(ROOT/'tables/primary_r18/per_image.csv', primary/'per_image.csv')
    env = dict(os.environ, AUDIT_ROOT=str(primary), PYTHONIOENCODING='utf-8')
    run = subprocess.run([sys.executable, str(ROOT/'analysis/summarize_primary.py')],
                         env=env, capture_output=True, text=True, encoding='utf-8')
    if run.returncode:
        raise RuntimeError(run.stderr)
    report = {'algorithm': 'Unchanged original summarize_primary.py',
              'scope': 'Derived CSV aggregation and existing image bootstrap only; no mask reevaluation',
              'stdout': run.stdout.strip(), 'comparisons': [], 'frozen_table_exports': []}
    for name in ['threshold_metrics.csv', 'density_summary.csv', 'density_differences.csv',
                 'paired_bootstrap.csv', 'budget_summary.csv']:
        expected = rows(ROOT/'tables/primary_r18'/name)
        actual = rows(primary/name)
        if len(expected) != len(actual):
            raise AssertionError(f'Row count mismatch: {name}')
        checked = 0
        for left, right in zip(expected, actual):
            if left.keys() != right.keys():
                raise AssertionError(f'Columns mismatch: {name}')
            for key in left:
                try:
                    a, b = float(left[key]), float(right[key])
                except ValueError:
                    if left[key] != right[key]:
                        raise AssertionError(f'Text mismatch: {name}/{key}')
                else:
                    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                        raise AssertionError(f'Numeric mismatch: {name}/{key}: {a} vs {b}')
                checked += 1
        report['comparisons'].append({'file': name, 'rows': len(actual), 'cells': checked,
                                      'status': 'PASS', 'relative_tolerance': 1e-12,
                                      'absolute_tolerance': 1e-12})
    exported = out/'frozen_tabular_exports'
    exported.mkdir()
    for source in sorted((ROOT/'tables/frozen_manuscript').glob('*_table_*.json')):
        data = json.loads(source.read_text(encoding='utf-8'))
        target = exported/(source.stem+'.tex')
        target.write_text(data['tabular']+'\n', encoding='utf-8')
        report['frozen_table_exports'].append({'file': target.name,
            'status': 'PASS', 'operation': 'Exact stored tabular transcription export; not numerical recomputation'})
    report['output_sha256'] = {p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(out.rglob('*')) if p.is_file()}
    report['status'] = 'PASS'
    (out/'TABLE_REPRODUCTION_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS: five primary summary CSVs reproduced and compared; '
          f'{len(report["frozen_table_exports"])} frozen table bodies exported.')


if __name__ == '__main__':
    main()
