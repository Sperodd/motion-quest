"""Shared server paths and streaming limits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_INTERVAL_MS = 250
MAX_MESSAGE_BYTES = 16_000
