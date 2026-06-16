"""Data layer: fetch OHLCV from multiple sources into a canonical Parquet store.

Crypto: ccxt (bitbank, live venue) + Binance public data (deep research history).
Equities: alpaca-py (free IEX / delayed SIP). Everything is normalized to one schema so
the research engine and executor never care where a bar came from.
"""
