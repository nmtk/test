"""The validation gate — the heart of the system.

A candidate becomes a challenger only after surviving, in order: combinatorial purged
cross-validation (CPCV), probability of backtest overfitting (PBO/CSCV), a deflated Sharpe
ratio that honestly accounts for the number of trials ever attempted, and a superior
predictive ability (SPA) check across the candidate family. Everything here is implemented
from primitives (numpy/pandas + stdlib) to avoid license/maintenance risk.
"""
