# Running IBM AMLSim for the CORAL benchmark

`run_amlsim.sh` reproduces the two AMLSim worlds used in the paper (held-out: random_seed 0; development: random_seed 1). Both have 10K-account parameter files, 8 typologies × 30 alerts, and 200 daily steps.

Changes relative to AMLSim commit 7338a4b. All are needed to run it today; none change normal behaviour:
1. **Java libraries.** MASON and org.json could not be downloaded, so `java_stubs/` provides minimal stand-ins for exactly the API AMLSim uses: `SimState`, `Schedule` (agents stepped in insertion order), `Steppable`, `MersenneTwisterFast` (as `java.util.Random` with bounded `nextLong`) and a small JSON parser. The WebGraph-based diameter statistic is stubbed out; it is disabled in the configs.
2. **Python generator.** It was written for networkx 1.11 and was ported to networkx 3 in `python_generator_nx3/`: node and edge attribute access, list-returning neighbour calls, `set_edge_attributes` argument order, `random.sample` on sets for Python ≥ 3.11, and an empty amount list for members without edges.
3. **Label fix (`amlsim_label_fix.patch`).** `BipartiteTypology` and `StackTypology` called the 4-argument `makeTransaction`, which marks transactions as normal (`isSAR=0, alertID=-1`). Their laundering transactions were therefore unlabeled. The patch passes `alert.isSAR(), alert.getAlertID()` as the other six typologies do. Re-simulation with the fix gives exactly the same 71,027 transactions; only the labels change.

Load into CORAL with `coral.data.load_amlsim(tx_log.csv, alert_members.csv)` or
`python scripts/run_loto.py --data amlsim --trans .../tx_log.csv --alerts .../alert_members.csv --window_days 7 --lookback 4 --sync_minutes 1440 --relay_hours 72`.
