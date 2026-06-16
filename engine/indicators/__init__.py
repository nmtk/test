"""Causal technical indicators (pandas/numpy only).

Every indicator at bar ``t`` uses information up to and including bar ``t``'s close and
never peeks ahead. The backtester is responsible for the one-bar execution delay (it
shifts target weights), so strategies may use the current bar's indicator value directly.
"""
