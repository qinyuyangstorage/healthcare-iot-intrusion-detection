from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_iomt(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Create non-clinical traffic with attack-like burst and timing patterns."""
    rng = np.random.default_rng(seed)
    label = rng.binomial(1, 0.48, size=n_rows)
    frame_len = rng.gamma(2.0, 55.0, size=n_rows) + label * rng.gamma(3.0, 120.0, size=n_rows)
    time_delta = rng.exponential(0.05, size=n_rows)
    time_delta += label * (0.16 + 0.08 * np.sin(np.arange(n_rows) / 9.0))
    tcp_retries = rng.poisson(0.15 + label * 2.2, size=n_rows)
    mqtt_gap = rng.normal(1.0, 0.08, size=n_rows) + label * np.sin(np.arange(n_rows) / 5.0)
    packet_rate = rng.normal(12, 2, size=n_rows) + label * rng.normal(18, 4, size=n_rows)

    data: dict[str, object] = {
        "frame.len": frame_len,
        "frame.time_delta": time_delta,
        "tcp.retries": tcp_retries,
        "mqtt.heartbeat_gap": mqtt_gap,
        "packet.rate": packet_rate,
        "protocol": np.where(rng.random(n_rows) > 0.45, "MQTT", "TCP"),
        "device.role": rng.choice(["monitor", "gateway", "environment"], n_rows),
        "ip.proto": rng.choice([6, 17], n_rows),
        "class": np.where(label == 1, "attack", "normal"),
        "label": label,
    }
    for index in range(17):
        data[f"network.feature_{index:02d}"] = rng.normal(label * (index % 3) * 0.2, 1.0, n_rows)
    return pd.DataFrame(data)
