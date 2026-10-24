"""Leakage tests. Run with:  python -m pytest tests  (or: python tests/test_leakage.py)

The core guarantee: a score for window t must not change if anything after
the end of window t changes."""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from coral.data import SyntheticConfig, generate_synthetic  # noqa: E402
from coral.evaluate import holdout, split_windows  # noqa: E402
from coral.model import CORAL, CORALConfig  # noqa: E402
from coral.windows import DAY, WindowBuilder  # noqa: E402

CFG = SyntheticConfig(n_accounts=600, n_days=16, patterns_per_typology=5, seed=3)


def _tx():
    return generate_synthetic(CFG)


def test_windows_never_read_the_future():
    b = WindowBuilder(_tx())
    for t in range(b.n_windows):
        w = b.build(t)
        assert w.max_ts_used < w.end, f"window {t} touched ts >= its end"
        assert (w.win_tx["ts"] >= w.start).all() and (w.win_tx["ts"] < w.end).all()


def test_scores_invariant_to_future_changes():
    tx = _tx()
    tr, va, te = split_windows(CFG.n_days)
    t = te[1]
    rng = np.random.default_rng(0)
    tx2 = tx.copy()
    fut = tx2["ts"] >= (t + 1) * DAY
    tx2.loc[fut, "amount"] = rng.permutation(tx2.loc[fut, "amount"].to_numpy()) * 3.0
    tx2.loc[fut, "dst"] = rng.permutation(tx2.loc[fut, "dst"].to_numpy())
    # (account ids are not re-indexed, so the id space N is unchanged)
    s1, _ = CORAL(CORALConfig(seed=1)).fit(WindowBuilder(tx), tr, va).score(WindowBuilder(tx).build(t))
    c2 = CORAL(CORALConfig(seed=1))
    b2 = WindowBuilder(tx2)
    s2, _ = c2.fit(b2, tr, va).score(b2.build(t))
    assert np.allclose(s1["coral"].to_numpy(), s2["coral"].to_numpy()), \
        "scores changed when only the future changed -> leakage"


def test_heldout_typology_absent_from_training():
    tx = _tx()
    tr, va, te = split_windows(CFG.n_days)
    txh = holdout(tx, "CYCLE", te[0] * DAY)
    b = WindowBuilder(txh)
    for t in tr + va:
        w = b.build(t)
        assert not (w.typology == "CYCLE").any()
        assert not (w.win_tx["typology"] == "CYCLE").any()
    assert any((b.build(t).typology == "CYCLE").any() for t in te), "test period must keep it"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
