"""Data loading for CORAL.

Every loader returns a transaction table with one canonical schema:

    ts            int64   seconds since dataset start
    src, dst      int64   account ids (contiguous, 0..n_accounts-1)
    amount        float64
    is_laundering int8    1 if the transaction is part of a laundering pattern
    pattern_id    int64   id of the laundering pattern (-1 for normal)
    typology      object  typology name, e.g. "FAN-OUT" ("" for normal)

Loaders:
    load_ibm_aml(trans_csv, patterns_txt)  IBM AMLworld (NeurIPS 2023, Kaggle)
    load_amlsim(tx_log, alert_members)     IBM AMLSim simulator output
    load_generic_csv(path, ...)            any account-to-account ledger
    generate_synthetic(...)                controllable synthetic world with decoys
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

TYPOLOGIES = ["FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER",
              "GATHER-SCATTER", "STACK", "BIPARTITE"]

SCHEMA = ["ts", "src", "dst", "amount", "is_laundering", "pattern_id", "typology"]


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = df["ts"].astype("int64")
    df["ts"] -= df["ts"].min()
    df["src"] = df["src"].astype("int64")
    df["dst"] = df["dst"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    df["is_laundering"] = df["is_laundering"].astype("int8")
    df["pattern_id"] = df["pattern_id"].astype("int64")
    df["typology"] = df["typology"].fillna("").astype(str)
    df = df[SCHEMA].sort_values("ts", kind="stable").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# IBM AMLworld
# --------------------------------------------------------------------------
_IBM_COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
             "Amount Received", "Receiving Currency", "Amount Paid",
             "Payment Currency", "Payment Format", "Is Laundering"]


def _parse_ibm_patterns(path: str) -> pd.DataFrame:
    """Parse the *_Patterns.txt file into rows with pattern_id and typology.

    The file contains blocks such as:
        BEGIN LAUNDERING ATTEMPT - FAN-OUT
        2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,...
        END LAUNDERING ATTEMPT - FAN-OUT
    Some headers carry extra text after a colon ("CYCLE:  Max 10 hops").
    """
    rows, pid, typ = [], -1, ""
    head = re.compile(r"BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Za-z\- ]+)")
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = head.match(line)
            if m:
                pid += 1
                typ = m.group(1).split(":")[0].strip().upper().replace(" ", "-")
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                continue
            parts = line.split(",")
            if len(parts) < 11:
                continue
            rows.append(parts[:11] + [pid, typ])
    pat = pd.DataFrame(rows, columns=_IBM_COLS + ["pattern_id", "typology"])
    return pat


def load_ibm_aml(trans_csv: str, patterns_txt: str | None = None,
                 nrows: int | None = None) -> pd.DataFrame:
    """Load an IBM AMLworld dataset (e.g. HI-Small_Trans.csv + HI-Small_Patterns.txt).

    Accounts are identified by (bank, account). Transactions from the patterns
    file are matched back to the ledger on (timestamp, from, to, amount paid)
    to attach pattern_id and typology. Laundering transactions that cannot be
    matched to a pattern keep typology "UNKNOWN" so they are never silently
    treated as normal.
    """
    raw = pd.read_csv(trans_csv, nrows=nrows, dtype=str)
    raw.columns = _IBM_COLS[: len(raw.columns)]
    raw["src_key"] = raw["From Bank"].str.strip() + ":" + raw["Account"].str.strip()
    raw["dst_key"] = raw["To Bank"].str.strip() + ":" + raw["Account.1"].str.strip()
    raw["ts_dt"] = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")
    raw["amt"] = pd.to_numeric(raw["Amount Paid"], errors="coerce").fillna(0.0)
    raw["is_laundering"] = pd.to_numeric(raw["Is Laundering"]).fillna(0).astype(int)

    raw["pattern_id"] = -1
    raw["typology"] = ""
    if patterns_txt:
        pat = _parse_ibm_patterns(patterns_txt)
        pat["src_key"] = pat["From Bank"].str.strip() + ":" + pat["Account"].str.strip()
        pat["dst_key"] = pat["To Bank"].str.strip() + ":" + pat["Account.1"].str.strip()
        pat["ts_dt"] = pd.to_datetime(pat["Timestamp"].str.strip(), format="%Y/%m/%d %H:%M")
        pat["amt"] = pd.to_numeric(pat["Amount Paid"], errors="coerce").round(2)
        key = ["ts_dt", "src_key", "dst_key", "amt"]
        raw["_amt_r"] = raw["amt"].round(2)
        pat = pat.drop_duplicates(key)[key + ["pattern_id", "typology"]]
        merged = raw.merge(pat.rename(columns={"amt": "_amt_r"}),
                           left_on=["ts_dt", "src_key", "dst_key", "_amt_r"],
                           right_on=["ts_dt", "src_key", "dst_key", "_amt_r"],
                           how="left", suffixes=("", "_p"))
        raw["pattern_id"] = merged["pattern_id_p"].fillna(-1).astype(int).values
        raw["typology"] = merged["typology_p"].fillna("").values
    unmatched = (raw["is_laundering"] == 1) & (raw["typology"] == "")
    raw.loc[unmatched, "typology"] = "UNKNOWN"

    keys = pd.Index(pd.unique(pd.concat([raw["src_key"], raw["dst_key"]])))
    df = pd.DataFrame({
        "ts": (raw["ts_dt"].astype("int64") // 10**9),
        "src": keys.get_indexer(raw["src_key"]),
        "dst": keys.get_indexer(raw["dst_key"]),
        "amount": raw["amt"],
        "is_laundering": raw["is_laundering"],
        "pattern_id": raw["pattern_id"],
        "typology": raw["typology"],
    })
    df = df[df["src"] != df["dst"]]  # self-transfers carry no relational signal
    return _finalize(df)


def load_amlsim(tx_log: str, alert_members: str) -> pd.DataFrame:
    """Load IBM AMLSim output (https://github.com/IBM/AMLSim): the simulator's tx_log.csv
    plus the generator's alert_members.csv, which gives each alert's typology
    (fan_in, fan_out, cycle, bipartite, stack, scatter_gather, gather_scatter, random).
    AMLSim time steps are days without a time of day, so ts = step * 86400."""
    t = pd.read_csv(tx_log)
    am = pd.read_csv(alert_members)
    reason = am.drop_duplicates("alertID").set_index("alertID")["reason"]
    typ = t["alertID"].map(reason).fillna("").astype(str).str.upper().str.replace("_", "-")
    df = pd.DataFrame({
        "ts": t["step"].astype("int64") * 86400,
        "src": t["nameOrig"].astype("int64"),
        "dst": t["nameDest"].astype("int64"),
        "amount": t["amount"].astype(float),
        "is_laundering": t["isSAR"].astype(int),
        "pattern_id": t["alertID"].where(t["isSAR"] == 1, -1).astype("int64"),
        "typology": typ.where(t["isSAR"] == 1, ""),
    })
    df.loc[(df["is_laundering"] == 1) & (df["typology"] == ""), "typology"] = "UNKNOWN"
    keys = pd.Index(pd.unique(pd.concat([df["src"], df["dst"]])))
    df["src"] = keys.get_indexer(df["src"]); df["dst"] = keys.get_indexer(df["dst"])
    df = df[df["src"] != df["dst"]]
    return _finalize(df)


def load_generic_csv(path: str, ts="ts", src="src", dst="dst", amount="amount",
                     label="is_laundering", pattern="pattern_id",
                     typology="typology", ts_format: str | None = None) -> pd.DataFrame:
    """Load any ledger (e.g. your own payment data). Pattern/typology columns are
    optional; without them only temporal (not typology-holdout) evaluation applies."""
    raw = pd.read_csv(path)
    t = raw[ts]
    if not np.issubdtype(t.dtype, np.number):
        t = pd.to_datetime(t, format=ts_format).astype("int64") // 10**9
    keys = pd.Index(pd.unique(pd.concat([raw[src].astype(str), raw[dst].astype(str)])))
    df = pd.DataFrame({
        "ts": t,
        "src": keys.get_indexer(raw[src].astype(str)),
        "dst": keys.get_indexer(raw[dst].astype(str)),
        "amount": raw[amount] if amount in raw else 1.0,
        "is_laundering": raw[label] if label in raw else 0,
        "pattern_id": raw[pattern] if pattern in raw else -1,
        "typology": raw[typology] if typology in raw else "",
    })
    df.loc[(df["is_laundering"] == 1) & (df["pattern_id"] < 0), "typology"] = "UNKNOWN"
    return _finalize(df)


# --------------------------------------------------------------------------
# Synthetic world
# --------------------------------------------------------------------------
@dataclass
class SyntheticConfig:
    n_accounts: int = 1500
    n_merchants: int = 30
    n_days: int = 28
    community_size: int = 40
    tx_per_account_day: float = 1.2
    p_random_counterparty: float = 0.12
    n_employers: int = 12          # legit recurring synchronized fan-out (payroll)
    group_splits_per_day: int = 8  # legit synchronized fan-in among friends
    novel_bursts_per_day: int = 2  # legit *novel* fan-in (e.g. a new seller)
    patterns_per_typology: int = 14
    frac_new_accounts: float = 0.5  # share of ring members that are fresh accounts
    frac_late_joiners: float = 0.15  # legit accounts that open during the period (churn)
    ring_amount_mu: float = 6.0     # log-mean of a pattern's total amount (overlaps normal)
    drift: bool = False             # stress test: misleading legit coordination in the last 30% of days
    drift_groups_per_day: int = 25
    seed: int = 7


def generate_synthetic(cfg: SyntheticConfig | None = None) -> pd.DataFrame:
    """A small world with realistic *decoys* so the task is not trivial:
    payroll (recurring synchronized fan-out), friend group-splits (synchronized
    fan-in with history), and novel legitimate bursts (new relationships).
    Laundering rings mix fresh and existing accounts and run in short bursts."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    DAY = 86400
    N, M = cfg.n_accounts, cfg.n_merchants
    merchants = np.arange(M)
    people = np.arange(M, N)
    comm = rng.integers(0, max(1, len(people) // cfg.community_size), size=N)
    # legit churn: some accounts open mid-period, so "new account" alone is not a fraud tell
    join = np.zeros(N, dtype=int)
    late = rng.random(len(people)) < cfg.frac_late_joiners
    join[people[late]] = rng.integers(1, cfg.n_days, size=late.sum())

    regulars = {}
    by_comm = pd.Series(people).groupby(comm[people]).apply(np.array).to_dict()
    for a in people:
        pool = by_comm[comm[a]]
        k = rng.integers(3, 9)
        same = rng.choice(pool, size=min(len(pool), max(1, int(k * 0.7))), replace=False)
        mer = rng.choice(merchants, size=max(1, k - len(same)), replace=False)
        regulars[a] = np.setdiff1d(np.concatenate([same, mer]), [a])

    rows = []  # ts, src, dst, amount, is_l, pid, typ

    def add(ts, s, d, amt, l=0, pid=-1, typ=""):
        if s != d:
            rows.append((int(ts), int(s), int(d), float(amt), l, pid, typ))

    employers = rng.choice(people, size=cfg.n_employers, replace=False)
    staff = {e: rng.choice(np.setdiff1d(people, [e]), size=rng.integers(8, 18), replace=False)
             for e in employers}

    for day in range(cfg.n_days):
        base = day * DAY
        live = people[join[people] <= day]
        live_all = np.r_[merchants, live]
        n_tx = rng.poisson(cfg.tx_per_account_day, size=len(live))
        n_tx[join[live] == day] += 2  # new accounts set up their usual payees on day one
        for a, k in zip(live, n_tx):
            reg = regulars[a][join[regulars[a]] <= day]
            for _ in range(k):
                if rng.random() < cfg.p_random_counterparty or len(reg) == 0:
                    d = rng.choice(live_all)
                else:
                    d = rng.choice(reg)
                t = base + int(np.clip(rng.normal(14, 4), 0, 23.9) * 3600)
                add(t, a, d, rng.lognormal(3.8, 1.0))
        # merchants pay suppliers occasionally
        for m in merchants:
            if rng.random() < 0.3:
                add(base + rng.integers(0, DAY), m, rng.choice(live), rng.lognormal(6, .5))
        # payroll: weekly, synchronized, recurring -> predictable from history
        if day % 7 == 4:
            for e in employers:
                t0 = base + 9 * 3600
                for s in staff[e]:
                    if join[s] <= day and join[e] <= day:
                        add(t0 + rng.integers(0, 600), e, s, rng.lognormal(7.2, .15))
        # friend group splits: synchronized fan-in among people with history
        for _ in range(cfg.group_splits_per_day):
            host = rng.choice(live)
            friends = regulars[host][(regulars[host] >= M) & (join[regulars[host]] <= day)]
            if len(friends) < 2:
                continue
            t0 = base + rng.integers(10 * 3600, 22 * 3600)
            for f in friends[: rng.integers(2, 6)]:
                add(t0 + rng.integers(0, 1800), f, host, rng.lognormal(3.2, .3))
        # novel legitimate bursts: new relationships, spread over the day
        for _ in range(cfg.novel_bursts_per_day):
            seller = rng.choice(live)
            for b in rng.choice(live, size=rng.integers(4, 10), replace=False):
                add(base + rng.integers(8 * 3600, 22 * 3600), b, seller, rng.lognormal(4.5, .4))

        # ---- drift stress test (off by default): a bill-splitting product launches
        # in the last 30% of days. Groups of strangers reimburse one payer within the
        # hour and the payer settles with a merchant: legitimate, but new, mutual,
        # synchronised and pass-through -- i.e. ring-like. Graph evidence becomes
        # misleading while account-level features change little.
        if cfg.drift and day >= int(0.7 * cfg.n_days):
            for _ in range(cfg.drift_groups_per_day):
                grp = rng.choice(live, size=rng.integers(4, 8), replace=False)
                payer, rest = grp[0], grp[1:]
                t0 = base + rng.integers(11 * 3600, 22 * 3600)
                share = rng.lognormal(3.4, .4)
                for f in rest:
                    add(t0 + rng.integers(0, 2400), f, payer, share * rng.uniform(.95, 1.05))
                for a_, b_ in zip(rest[:-1], rest[1:]):   # friends settle among themselves too
                    if rng.random() < .5:
                        add(t0 + rng.integers(0, 2400), a_, b_, share * rng.uniform(.3, .6))
                add(t0 + rng.integers(2400, 3600), payer, rng.choice(merchants), share * len(grp))

    # ---- laundering patterns -------------------------------------------
    next_id = [N]
    pid = 0

    def members(k, day):
        pool = people[join[people] <= day]
        out = []
        for _ in range(k):
            if rng.random() < cfg.frac_new_accounts:
                out.append(next_id[0]); next_id[0] += 1
            else:
                out.append(int(rng.choice(pool)))
        return out

    for typ in TYPOLOGIES:
        days = rng.integers(2, cfg.n_days, size=cfg.patterns_per_typology)
        for day in days:
            t = day * DAY + rng.integers(0, 18 * 3600)
            amt = rng.lognormal(cfg.ring_amount_mu, .8)
            step = lambda: rng.integers(5 * 60, 50 * 60)  # noqa: E731
            k = int(rng.integers(4, 8))
            if typ == "FAN-OUT":
                s, *ts_ = members(1 + k, day)
                for d in ts_:
                    t += step(); add(t, s, d, amt / k * rng.uniform(.9, 1), 1, pid, typ)
            elif typ == "FAN-IN":
                d, *ss = members(1 + k, day)
                for s in ss:
                    t += step(); add(t, s, d, amt / k * rng.uniform(.9, 1), 1, pid, typ)
            elif typ == "CYCLE":
                ring = members(k, day)
                for i in range(k):
                    t += step(); amt *= .98
                    add(t, ring[i], ring[(i + 1) % k], amt, 1, pid, typ)
            elif typ == "SCATTER-GATHER":
                s, d, *mid = members(2 + k, day)
                for m_ in mid:
                    t += step(); add(t, s, m_, amt / k, 1, pid, typ)
                for m_ in mid:
                    t += step(); add(t, m_, d, amt / k * .97, 1, pid, typ)
            elif typ == "GATHER-SCATTER":
                mid, *rest = members(1 + 2 * (k // 2 + 1), day)
                h = len(rest) // 2
                for s in rest[:h]:
                    t += step(); add(t, s, mid, amt / h, 1, pid, typ)
                for d in rest[h:]:
                    t += step(); add(t, mid, d, amt / h * .97, 1, pid, typ)
            elif typ == "STACK":
                w = max(2, k // 2)
                L = [members(w, day) for _ in range(3)]
                for a_, b_ in ((L[0], L[1]), (L[1], L[2])):
                    for s in a_:
                        for d in b_:
                            if rng.random() < .7:
                                t += step() // 3; add(t, s, d, amt / (w * w), 1, pid, typ)
            elif typ == "BIPARTITE":
                w = max(2, k // 2)
                A, B = members(w, day), members(w, day)
                for s in A:
                    for d in B:
                        if rng.random() < .6:
                            t += step() // 3; add(t, s, d, amt / (w * w), 1, pid, typ)
            pid += 1

    df = pd.DataFrame(rows, columns=SCHEMA)
    return _finalize(df)
