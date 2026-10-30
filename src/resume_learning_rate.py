"""Synchronize PPO's adaptive learning-rate variable with restored Adam state (MIT)."""
import math


def restore_algorithm_learning_rate(algorithm):
    rates = [float(group['lr']) for group in algorithm.optimizer.param_groups]
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError('Restored optimizer learning rates must be finite and positive')
    if any(rate != rates[0] for rate in rates[1:]):
        raise ValueError('A scalar PPO schedule cannot preserve different optimizer group rates')
    before = float(algorithm.learning_rate)
    algorithm.learning_rate = rates[0]
    return dict(algorithm_before=before, algorithm_after=algorithm.learning_rate,
                optimizer_rates=rates, optimizer_unchanged=True)
