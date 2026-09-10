from pathlib import Path
import numpy as np
from django.test import TestCase
from django.urls import reverse

from .services.data_service import load_movies
from .services.feature_service import extract_features
from .services.preference_model import fit_bradley_terry, fit_plackett_luce
from .services.study_service import (
    PAIRWISE_TASKS,
    RANKING_SIZE,
    RANKING_TASKS,
    VALIDATION_TASKS,
    build_plan,
)


class PreferenceModelTests(TestCase):
    def test_pairwise_fit_prefers_first_feature(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
        w = fit_bradley_terry(X, [(0, 1), (2, 1)])
        self.assertGreater(w[0], w[1])

    def test_ranking_fit_orders_items(self):
        X = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        w = fit_plackett_luce(X, [[0, 1, 2]])
        self.assertGreater(X[0] @ w, X[2] @ w)


class FeatureAndPlanTests(TestCase):
    def test_feature_matrix_is_finite_and_matches_movie_count(self):
        df = load_movies()
        X, names = extract_features(df)
        self.assertEqual(X.shape[0], len(df))
        self.assertEqual(X.shape[1], len(names))
        self.assertTrue(np.isfinite(X).all())

    def test_study_plan_has_expected_sizes_and_no_movie_overlap(self):
        plan = build_plan(500, seed=123)
        self.assertEqual(len(plan['pairs']), PAIRWISE_TASKS)
        self.assertTrue(all(len(pair) == 2 for pair in plan['pairs']))
        self.assertEqual(len(plan['rankings']), RANKING_TASKS)
        self.assertTrue(all(len(ranking) == RANKING_SIZE for ranking in plan['rankings']))
        self.assertEqual(len(plan['validation']), VALIDATION_TASKS)

        all_ids = (
            sum(plan['pairs'], [])
            + sum(plan['rankings'], [])
            + sum(plan['validation'], [])
        )
        self.assertEqual(len(all_ids), len(set(all_ids)))


class InterfaceRequirementTests(TestCase):
    def test_landing_page_has_start_and_report_actions(self):
        response = self.client.get(reverse('project4:landing'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Start the study')
        self.assertContains(response, 'Download PDF report')
        self.assertContains(response, 'Task 1')
        self.assertContains(response, 'Task 4')

    def test_report_is_downloadable_pdf(self):
        response = self.client.get(reverse('project4:report'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')

    def test_static_report_file_is_present_and_is_pdf(self):
        report_path = Path(__file__).resolve().parent / 'HCAI_Project_4_Report.pdf'
        self.assertTrue(report_path.exists())
        self.assertTrue(report_path.read_bytes().startswith(b'%PDF'))
        self.assertGreater(report_path.stat().st_size, 5000)
