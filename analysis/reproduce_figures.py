"""Run existing CSV-driven figure generators in a disposable working copy."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError('Use an output directory outside the immutable release.')
    out.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Fig. 1, Fig. 3 and Fig. S1 only; qualitative figure requires external data',
              'algorithm': 'Existing generators and input CSVs unchanged', 'runs': []}
    with tempfile.TemporaryDirectory(prefix='query_budget_figures_') as temp:
        work = Path(temp)
        for folder, script in [('four_budget','plot_saved_analysis.py'),
                               ('tradeoff','make_query_tradeoff_figure.py')]:
            shutil.copytree(ROOT/'figures'/folder,work/folder)
            env=dict(os.environ,MPLBACKEND='Agg',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
            result=subprocess.run([sys.executable,str(work/folder/script)],env=env,
                capture_output=True,text=True,encoding='utf-8')
            if result.returncode:
                raise RuntimeError(result.stderr)
            # Traceback/local paths are deliberately not persisted in public test metadata.
            report['runs'].append({'script':f'figures/{folder}/{script}', 'exit_code':0,
                'alignment_adapter_warning': 'External alignment-only QA is not bundled; visual check required'})
        for name, source in [
            ('fig_four_budget_stratified_response',work/'four_budget/figures/four_budget_stratified_response'),
            ('four_budget_threshold_pr',work/'four_budget/figures/four_budget_threshold_pr'),
            ('fig_query_tradeoff',work/'tradeoff/fig_query_tradeoff')]:
            for suffix in ['.pdf','.png']:
                shutil.copy2(source.with_suffix(suffix),out/(name+suffix))
    report['status']='PASS'
    report['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(out.iterdir()) if p.is_file()}
    (out/'FIGURE_REPRODUCTION_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS: three figures regenerated from included CSVs; six PDF/PNG outputs.')


if __name__ == '__main__':
    main()
