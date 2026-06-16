"""Execution layer: a common broker interface with bitbank, Alpaca, and paper adapters.

The engine speaks only the abstract :class:`~engine.execution.broker.Broker` /
:class:`~engine.execution.broker.MarketData` interfaces; venue-specific code lives behind
adapters. Live adapters never call withdrawal endpoints.
"""
