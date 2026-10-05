"""F2 numerical references plus the unchanged contextual integration suites."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from app.modulos.contextual.metrics import (
    signed_association, nominal_association, mixed_information, numeric_dtw, library_versions,
)


def pearson(x, y):
    mx, my = sum(x)/len(x), sum(y)/len(y)
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/math.sqrt(
        sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))


class NumericalReferences(unittest.TestCase):
    def test_pearson_official_example_and_equation(self):
        x, y = [1, 2, 3, 4, 5, 6, 7], [10, 9, 2.5, 6, 4, 3, 2]
        actual = signed_association(pd.DataFrame({'source': x, 'target': y}), 'pearson')
        self.assertAlmostEqual(actual, -.828503883588428)
        self.assertAlmostEqual(actual, pearson(x, y))

    def test_spearman_official_tied_rank_example(self):
        pair = pd.DataFrame({'source': [1, 2, 3, 4, 5], 'target': [5, 6, 7, 8, 7]})
        actual = signed_association(pair, 'spearman')
        self.assertAlmostEqual(actual, .8207826816681233)
        self.assertAlmostEqual(actual, pearson([1, 2, 3, 4, 5], [1, 2, 3.5, 5, 3.5]))

    def test_nominal_against_contingency_equations(self):
        for counts in ([[10, 0], [0, 10]], [[5, 5], [5, 5]], [[12, 3], [4, 11]]):
            source, target = [], []
            for i in range(2):
                for j in range(2):
                    source += [f's{i}']*counts[i][j]
                    target += [f't{j}']*counts[i][j]
            actual = nominal_association(pd.Series(source), pd.Series(target))
            n = sum(map(sum, counts))
            rows = [sum(row) for row in counts]
            cols = [sum(counts[i][j] for i in range(2)) for j in range(2)]
            chi2, mi = 0., 0.
            for i in range(2):
                for j in range(2):
                    expected = rows[i]*cols[j]/n
                    chi2 += (counts[i][j]-expected)**2/expected
                    if counts[i][j]:
                        mi += counts[i][j]/n*math.log(counts[i][j]*n/(rows[i]*cols[j]))
            self.assertAlmostEqual(actual['cramers_v'], math.sqrt(chi2/n))
            self.assertAlmostEqual(actual['mutual_information_nats'], mi)

    def test_dtw_against_independent_dynamic_programming(self):
        # Full band, no resampling: isolate the documented DTW recurrence.
        x, y = [0., 1., 2., 3., 4.], [0., 1., 1., 3., 5.]
        def normalize(values):
            mean = sum(values)/len(values)
            scale = math.sqrt(sum((v-mean)**2 for v in values)/len(values))
            centre = sorted(values)[len(values)//2]
            return [(v-centre)/scale for v in values]
        a, b = normalize(x), normalize(y)
        cost = [[math.inf]*(len(b)+1) for _ in range(len(a)+1)]
        cost[0][0] = 0.
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                cost[i][j] = (a[i-1]-b[j-1])**2 + min(cost[i-1][j], cost[i][j-1], cost[i-1][j-1])
        expected = math.sqrt(cost[-1][-1]/len(a))
        actual = numeric_dtw(pd.DataFrame({'source': x, 'target': y}), band_fraction=1.)
        self.assertAlmostEqual(actual['distance'], expected)
        self.assertAlmostEqual(actual['similarity'], 1/(1+expected))

    def test_mixed_wrapper_matches_documented_discrete_feature_api(self):
        # Integration parity, not an independent proof of the kNN estimator.
        numeric = pd.Series(np.r_[np.arange(20.)*.01, 10+np.arange(20.)*.01])
        categorical = pd.Series(['idle']*20+['active']*20)
        codes = np.r_[np.zeros(20), np.ones(20)].reshape(-1, 1)
        reference = mutual_info_regression(codes, numeric.to_numpy(), discrete_features=True,
                                           n_neighbors=3, random_state=42)[0]
        forward = mixed_information(numeric, categorical, 'continuous')
        reverse = mixed_information(categorical, numeric, 'multistate')
        self.assertAlmostEqual(forward['mutual_information_nats'], reference)
        self.assertEqual(forward, reverse)

    def test_invalid_and_constant_pairs_not_evidence(self):
        pair = pd.DataFrame({'source': [0., 1., np.nan, np.inf], 'target': [1., 3., 4., 5.]})
        self.assertAlmostEqual(signed_association(pair), 1.)
        pair['target'] = 3.
        self.assertIsNone(signed_association(pair))
        self.assertIsNone(numeric_dtw(pair))
        self.assertIsNone(nominal_association(pd.Series(['a']*4), pd.Series(['b']*4))['cramers_v'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'tests/backend'))
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(NumericalReferences),
        unittest.defaultTestLoader.discover(str(root/'tests/backend'), pattern='test_contextual*.py'),
    ])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {'tests_run': result.testsRun, 'success': result.wasSuccessful(),
              'failures': [{'test': str(t), 'traceback': tb} for t, tb in result.failures],
              'errors': [{'test': str(t), 'traceback': tb} for t, tb in result.errors],
              'versions': library_versions(), 'log': stream.getvalue(),
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (root/'backend/app/modulos/contextual').glob('*.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(stream.getvalue())
    print(json.dumps(report['versions']))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
