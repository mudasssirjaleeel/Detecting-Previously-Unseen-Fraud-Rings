#!/usr/bin/env bash
# Reproduce the AMLSim datasets used in the paper.
# Requires: git, Java 8+ (javac), Python 3 with numpy + networkx 3.x.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
git clone https://github.com/IBM/AMLSim amlsim && cd amlsim
git checkout 7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca
git apply "$HERE/amlsim_label_fix.patch"                 # label bipartite/stack transactions as SAR
cp -r "$HERE/paramFiles_coral10k" paramFiles/coral10k
cp "$HERE"/conf_coral10k*.json .
# Compile with minimal stand-ins for MASON (SimState, Schedule, Steppable, MersenneTwisterFast),
# org.json and the optional WebGraph diameter statistic (disabled in our configs).
find src/main/java -name "*.java" ! -path "*/stat/Diameter.java" > srcs.txt
find "$HERE/java_stubs" -name "*.java" >> srcs.txt
mkdir -p build && javac -nowarn -d build @srcs.txt
for c in conf_coral10k conf_coral10k_dev; do
  python3 "$HERE/python_generator_nx3/transaction_graph_generator.py" $c.json   # networkx-3 port of the generator
  java -Xmx4g -cp build amlsim.AMLSim $c.json
done
echo "Held-out world: outputs/coral10k/tx_log.csv + tmp/coral10k/alert_members.csv"
echo "Development world: outputs/coral10k_dev/tx_log.csv + tmp/coral10k_dev/alert_members.csv"
