#!/usr/bin/env python3
"""Check that all reference candidates completed; speed is reported, never assumed."""
import argparse
import json
import math
from pathlib import Path
import statistics

REFERENCE_PATHS = {'bf16': 'torch_mm', 'fp16': 'torch_mm_then_bf16', 'fp8': 'aiter_gfx1201_mm8'}


def verify(actual, reference):
    expected = {(r['precision'], r['candidate']) for r in reference['results']}
    rows = actual.get('results', [])
    errors = []
    if len(rows) != len(expected) or {(r['precision'], r['candidate']) for r in rows} != expected:
        errors.append('Incomplete reference candidate coverage')
    parameters = actual.get('metadata', {}).get('parameters', {})
    for key in ('warmup', 'samples', 'repeats', 'max_nrmse', 'tune_warmup', 'tune_samples',
                'torch_tune_duration_ms', 'torch_tune_iterations', 'skip_tune'):
        if parameters.get(key) != reference['metadata']['parameters'][key]:
            errors.append('Historical parameter differs: ' + key)
    if not actual.get('metadata', {}).get('tuning_search_complete'):
        errors.append('Tuning phase incomplete')
    comparison = {}
    for row in rows:
        name = f"{row['precision']}/{row['candidate']}"
        values = row.get('samples_ms', [])
        if (row.get('status') != 'ok' or len(values) != 11
                or not all(math.isfinite(v) and v > 0 for v in values)
                or not math.isclose(statistics.median(values), row.get('median_ms', -1), rel_tol=1e-9)
                or not math.isfinite(row.get('nrmse', float('inf')))
                or row.get('nrmse', 1) > 0.02
                or not math.isfinite(row.get('max_abs', float('inf')))
                or row.get('max_abs', float('inf')) > row.get('max_abs_limit', -1)):
            errors.append('Unqualified result: ' + name)
            continue
        p = row['precision']
        if row['candidate'] == REFERENCE_PATHS[p]:
            old = next(r for r in reference['results'] if r['precision'] == p and r['candidate'] == row['candidate'])
            comparison[p] = {'implementation': row['candidate'], 'measured_ms': row['median_ms'],
                             'measured_tflops': 1099.511627776 / row['median_ms'],
                             'reference_ms': old['median_ms'], 'reference_tflops': old['tflops']}
    return {'complete': not errors, 'errors': errors, 'gpu': actual.get('metadata', {}).get('gpu'),
            'same_gpu_model_as_reference': actual.get('metadata', {}).get('gpu') == reference['metadata']['gpu'],
            'comparison': comparison, 'note': 'Complete means numerical/measurement checks passed; it is not a latency guarantee.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.results.read_text()), json.loads(args.reference.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['complete'] else 2)
