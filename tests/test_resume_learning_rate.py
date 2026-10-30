"""Regression tests for continuation-rate restoration; no GPU required (MIT)."""
from types import SimpleNamespace
import unittest
from resume_learning_rate import restore_algorithm_learning_rate


class ResumeRateTest(unittest.TestCase):
    def algorithm(self, rates):
        return SimpleNamespace(learning_rate=.001,optimizer=SimpleNamespace(param_groups=[{'lr':r} for r in rates]))

    def test_restores_scalar_without_mutating_optimizer(self):
        algorithm=self.algorithm([.00017])
        groups=algorithm.optimizer.param_groups
        record=restore_algorithm_learning_rate(algorithm)
        self.assertEqual(algorithm.learning_rate,.00017)
        self.assertIs(algorithm.optimizer.param_groups,groups)
        self.assertEqual(groups,[{'lr':.00017}])
        self.assertEqual(record['algorithm_before'],.001)

    def test_rejects_ambiguous_and_invalid_rates_before_mutation(self):
        for rates in [[],[0],[-1],[float('nan')],[float('inf')],[.001,.002]]:
            with self.subTest(rates=rates):
                algorithm=self.algorithm(rates)
                with self.assertRaises(ValueError):restore_algorithm_learning_rate(algorithm)
                self.assertEqual(algorithm.learning_rate,.001)


if __name__=='__main__':unittest.main()
