"""Small metrics facade.

Business code can depend on this module without knowing whether the current
process exports metrics. In ``local`` and ``off`` modes these functions are
cheap no-ops.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Mapping, Optional

from .config import ObservabilityConfig

logger = logging.getLogger(__name__)

_config: Optional[ObservabilityConfig] = None
_counters: dict[str, object] = {}
_histograms: dict[str, object] = {}


def configure_metrics(config: ObservabilityConfig) -> None:
    global _config
    _config = config


@lru_cache(maxsize=1)
def _meter():
    try:
        from opentelemetry import metrics
    except ImportError:
        return None
    return metrics.get_meter("aperag")


def get_meter():
    return _meter()


def record_counter(name: str, value: int = 1, attributes: Mapping[str, object] | None = None) -> None:
    if _config is None or not _config.metrics_enabled:
        return
    meter = _meter()
    if meter is None:
        return
    try:
        counter = _counters.get(name)
        if counter is None:
            counter = meter.create_counter(name)
            _counters[name] = counter
        counter.add(value, dict(attributes or {}))
    except Exception as exc:
        logger.debug("Failed to record counter %s: %s", name, exc)


def record_histogram(name: str, value: float, attributes: Mapping[str, object] | None = None) -> None:
    if _config is None or not _config.metrics_enabled:
        return
    meter = _meter()
    if meter is None:
        return
    try:
        histogram = _histograms.get(name)
        if histogram is None:
            histogram = meter.create_histogram(name)
            _histograms[name] = histogram
        histogram.record(value, dict(attributes or {}))
    except Exception as exc:
        logger.debug("Failed to record histogram %s: %s", name, exc)
