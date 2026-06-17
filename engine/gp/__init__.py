"""Genetic-programming alpha discovery.

Evolves *new* alpha formulas from a set of causal operators, scored by the same robust
out-of-sample objective the parameter search uses. Discovered programs are wrapped as a
``Strategy`` (``gp``) and flow through the identical validation gate and incubation guardrail
as every other candidate — the generator is powerful, but it is inside the cage (M5).
"""
