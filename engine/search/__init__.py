"""Strategy search: optuna parameter optimization feeding the validation gate.

The search proposes parameters, scores each by *robust* out-of-sample performance (not raw
backtest Sharpe), then runs the explored family through the M2 gate. Only a family that
clears the gate yields a challenger. The search counts toward the trial ledger, so the
Deflated Sharpe Ratio is deflated by every parameter ever tried.
"""
