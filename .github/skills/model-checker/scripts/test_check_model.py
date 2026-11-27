"""Regression tests for check_model WWB NaN metric validation."""

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_model as cm


class TestParseWWBMetricsValue(unittest.TestCase):
    def test_finite_value(self):
        self.assertEqual(cm.parse_wwb_metrics_value(["x", "y", "0    0.97"]), 0.97)

    def test_nan_value(self):
        v = cm.parse_wwb_metrics_value(["x", "y", "0    nan"])
        self.assertNotEqual(v, v)

    def test_none_on_empty(self):
        self.assertIsNone(cm.parse_wwb_metrics_value([]))


class TestPostRunHookNonFinite(unittest.TestCase):
    def _make_result(self, stdout):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    def test_nan_fails_optimum(self):
        tool = cm.OptimumWWBTargetEvaluationTool.__new__(cm.OptimumWWBTargetEvaluationTool)
        tool.name = "wwb_optimum_target_eval"
        tool.logger_prefix = "optimum"
        tool.work_dir = Path("/tmp")
        # bypass __init__ command setup; _post_run_hook only uses attributes above + self.name
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    nan"]))
        self.assertFalse(r.success)
        self.assertIn("similarity nan is not finite", r.err_msg)

    def test_nan_fails_genai(self):
        tool = cm.GenAIWWBTargetEvaluationTool.__new__(cm.GenAIWWBTargetEvaluationTool)
        tool.name = "wwb_genai_target_eval"
        tool.logger_prefix = "genai"
        tool.work_dir = Path("/tmp")
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    nan"]))
        self.assertFalse(r.success)
        self.assertIn("similarity nan is not finite", r.err_msg)

    def test_pos_inf_fails_optimum(self):
        tool = cm.OptimumWWBTargetEvaluationTool.__new__(cm.OptimumWWBTargetEvaluationTool)
        tool.name = "wwb_optimum_target_eval"
        tool.logger_prefix = "optimum"
        tool.work_dir = Path("/tmp")
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    inf"]))
        self.assertFalse(r.success)
        self.assertIn("similarity inf is not finite", r.err_msg)

    def test_neg_inf_fails_optimum(self):
        tool = cm.OptimumWWBTargetEvaluationTool.__new__(cm.OptimumWWBTargetEvaluationTool)
        tool.name = "wwb_optimum_target_eval"
        tool.logger_prefix = "optimum"
        tool.work_dir = Path("/tmp")
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    -inf"]))
        self.assertFalse(r.success)
        self.assertIn("similarity -inf is not finite", r.err_msg)

    def test_pos_inf_fails_genai(self):
        tool = cm.GenAIWWBTargetEvaluationTool.__new__(cm.GenAIWWBTargetEvaluationTool)
        tool.name = "wwb_genai_target_eval"
        tool.logger_prefix = "genai"
        tool.work_dir = Path("/tmp")
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    inf"]))
        self.assertFalse(r.success)
        self.assertIn("similarity inf is not finite", r.err_msg)

    def test_neg_inf_fails_genai(self):
        tool = cm.GenAIWWBTargetEvaluationTool.__new__(cm.GenAIWWBTargetEvaluationTool)
        tool.name = "wwb_genai_target_eval"
        tool.logger_prefix = "genai"
        tool.work_dir = Path("/tmp")
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    -inf"]))
        self.assertFalse(r.success)
        self.assertIn("similarity -inf is not finite", r.err_msg)

    def test_finite_passing(self):
        tool = cm.OptimumWWBTargetEvaluationTool.__new__(cm.OptimumWWBTargetEvaluationTool)
        tool.name = "wwb_optimum_target_eval"
        tool.logger_prefix = "optimum"
        tool.work_dir = Path("/tmp")
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    0.97"]))
        self.assertTrue(r.success)

    def test_finite_below_threshold_fails(self):
        tool = cm.OptimumWWBTargetEvaluationTool.__new__(cm.OptimumWWBTargetEvaluationTool)
        tool.name = "wwb_optimum_target_eval"
        tool.logger_prefix = "optimum"
        tool.work_dir = Path("/tmp")
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    0.90"]))
        self.assertFalse(r.success)
        self.assertIn("below threshold", r.err_msg)

    def test_finite_passing_genai(self):
        tool = cm.GenAIWWBTargetEvaluationTool.__new__(cm.GenAIWWBTargetEvaluationTool)
        tool.name = "wwb_genai_target_eval"
        tool.logger_prefix = "genai"
        tool.work_dir = Path("/tmp")
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    0.97"]))
        self.assertTrue(r.success)

    def test_finite_below_threshold_fails_genai(self):
        tool = cm.GenAIWWBTargetEvaluationTool.__new__(cm.GenAIWWBTargetEvaluationTool)
        tool.name = "wwb_genai_target_eval"
        tool.logger_prefix = "genai"
        tool.work_dir = Path("/tmp")
        r = tool._post_run_hook(self._make_result(["INFO", "similarity", "0    0.90"]))
        self.assertFalse(r.success)
        self.assertIn("below threshold", r.err_msg)


if __name__ == "__main__":
    unittest.main()
