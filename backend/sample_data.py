import numpy as np
import pandas as pd


def make_sample_csv_bytes():
    """The built-in demo data: 300 made-up shop orders, with a few blank regions on purpose (like real data)."""
    rs = np.random.RandomState(42)               # fixed seed: everyone gets the same sample
    n = 300
    frame = pd.DataFrame({
        "order_id": range(1, n + 1),
        "order_date": pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "region": rs.choice(["North", "South", "East", "West"], n),
        "category": rs.choice(["Shoes", "Bags", "Belts", "Wallets"], n),
        "quantity": rs.randint(1, 10, n),
        "unit_price": rs.randint(300, 3000, n),
    })
    frame["revenue"] = frame["quantity"] * frame["unit_price"]
    frame.loc[rs.choice(n, 10, replace=False), "region"] = None
    return frame.to_csv(index=False).encode("utf-8")