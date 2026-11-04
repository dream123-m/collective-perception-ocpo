# Reproduction materials: collective perception with MAPPO and OCPO

This folder contains the adopted MAPPO, OCPO and MADDPG implementations,
archived vehicle layouts, selected checkpoints, raw experimental records,
and data for the revised experimental figures and tables. Run commands
from this folder. No access to the author's working directory is required.

## Core code

The OCPO algorithm is in `OCPO/update.py`: overlap-aware decision factors,
factor log probabilities, and the clipped team-advantage policy loss. The
training and evaluation runner is `major/cross_scene/run.py`; select OCPO with
`--method ocpo`. `major/model.py` defines the shared policy network.
`MAPPO/` contains the comparison network definition and archived vehicle
scenes. The repository includes both OCPO and MAPPO so the reported comparison
can be reproduced.

## Quick reproduction

Use Python 3.12 and the dependencies in requirements.txt:

```text
python -m pip install -r requirements.txt
python -B plot_figures.py --data-dir data --output-dir outputs/figures
```

The second command draws experimental Figures 5–9 directly from the supplied
CSV/JSON files. It does not load models, rerun experiments or require the
archived training folders. PDF panels and PNG previews go to outputs/figures/.
`python -B review.py plot` draws the same figures using the default paths.

For data and model checks and one model-based example, run
`python -B review.py all`. This does not start a full training run.

For Codex, open this folder and use this instruction:

> Install the dependencies in requirements.txt if needed. Run
> `python -B review.py all`, then `python -B review.py evaluate --scope main`.
> Report whether the checks pass. Show the generated figures and table values.
> Preserve the supplied raw data, model files, source code, and statistics.
> If a command fails, report the error before making any changes.

The commands also work directly without Codex. Plotting uses the same colors,
panel sizes and smoothing as the manuscript. Arial and Times New Roman are
used when installed; a system without these fonts may substitute its fonts.
Figures 6 and 7 are exported as separate panels for assembly in LaTeX.

## Data and paper mapping

| Paper item | Supplied evidence | Command |
|---|---|---|
| Fig. 5, training rewards | data/fig5_training.csv | `python -B plot_figures.py --figure 5 --data-dir data` |
| Fig. 6, payload at target 0.90 | data/fig6_7_summary.csv | `python -B plot_figures.py --figure 6 --data-dir data` |
| Fig. 7, payload at target 0.95 | data/fig6_7_summary.csv | `python -B plot_figures.py --figure 7 --data-dir data` |
| Fig. 8, mobility | data/fig8_summary.json (raw records: data/fig8_per_layout.csv) | `python -B plot_figures.py --figure 8 --data-dir data` |
| Fig. 9, prescribed coverage target | data/fig9_per_cav.csv and data/fig9_summary.json | `python -B plot_figures.py --figure 9 --data-dir data` |
| Table IV, rewards and components | data/table4_per_seed.csv and table4.csv; selected models and per-layout tests | `python -B review.py evaluate --scope ablation` |
| Table V, inference latency | data/table5_samples.csv, table5_per_seed.csv, table5_summary.json; scenario MAT files | `python -B review.py latency` |
| Table VI, spatial layouts | data/table6_per_seed.csv, table6_summary.json; the baseline and headway MAT files | Source: major/cross_scene/legacy_spatial.py |

Model-based replay is separate from plotting: use `review.py evaluate --scope
grid` for Figures 6–7 or `review.py evaluate --scope target` for Figure 9.

The main table uses seeds 42, 123 and 456, with mean and sample standard
deviation over training runs. The scenario grid uses seed 123 and averages
each condition over 32 test layouts. A layout is not treated as a training
seed. Fig. 5 smooths each run with a centered 20-episode moving mean, then
plots the three-run mean and sample standard deviation. Fig. 9 points are
CAV-role averages over layouts, not independent training runs.

## Reviewer Comment 1.2: source, raw data and a runnable example

```text
python -B review.py example
python -B review.py evaluate --scope main
python -B review.py smoke
```

The example loads the selected OCPO checkpoint for seed 123, uses the first
test layout, executes the recorded seven decisions and compares reward,
coverage and payload with its supplied test record. Main evaluation repeats
the 32-layout tests of all three methods and all three training seeds.
The tolerance for replayed metrics is 1e-6.

The short check runs the actual training code on temporary outputs, including
the adopted MADDPG update. It is a runtime check, separate from the recorded
1500-episode results. Full training can be started with:

```text
python -u -B review.py train --method ocpo --seed 123
python -u -B review.py train --method mappo --seed 123
python -u -B review.py train --method maddpg --seed 123
```

Each command trains from initialization for 1500 episodes with the archived
validation rule. Use --seed 42 or --seed 456 for the other main runs. Scenario
arguments are --density, --penetration and --target. New training outputs are
written under major/cross_scene/results/new_training/.

The MADDPG baseline uses binary ST Gumbel softmax actions, local actors and
one shared team critic. The critic receives the MAPPO global-state fields
and joint actions. All three methods use reward coefficients
(0.2, 0.6, 0.2, 1.4, 6.0) and coverage reward width 0.04. The archived run
records contain the network and optimizer configurations. The adopted
MADDPG main reward is 3.5240 ± 0.2568; MAPPO and OCPO are
4.8272 ± 0.0846 and 4.9198 ± 0.1507, respectively.

## Reviewer Comment 1.3: inference scalability data and construction

The five source scenes contain 27, 41, 65, 101 and 107 CAVs. They are the
included MAT files under MAPPO/; the exact paths, vehicle counts, CAV counts,
and hashes are recorded in
major/cross_scene/results/legacy_vehicle/inference_latency/protocol.json.
The timing source reconstructs visibility from each scene and creates local
actors with scenario-specific input and output dimensions.

For each size, three initializations are measured with one CPU thread,
20 warmups and 200 measured repetitions per initialization. The 65-CAV
case also uses the three trained OCPO policies. The 3600 individual measured
times and their per-run means are supplied, so the reported aggregation can
be checked without repeating a benchmark. A new benchmark measures inference
on the reviewer's hardware and writes its own samples to outputs/latency/.
Elapsed times depend on hardware and system load.

## Layouts and models

data/datasets.json lists each layout bank. Each bank contains 64 training,
16 validation and 32 test layouts. Positions vary with scenario-specific
vehicle composition and CAV roles; complete layout geometries are disjoint
between splits. Original layout manifests retain the generator settings,
seeds and file hashes. Layout generation and simulation source are included.

The simulation recomputes sensing geometry and communication relationships
from vehicle positions. The archived comparison uses a fixed underlying
fading draw. Coverage counts objects selected by a sender with a decoding
receiver; cochannel interference is supplied by the other CAVs independently
of their selected report sets. These settings are common to the compared
methods. Their exact configuration is recorded with every run.

models.json maps every supplied model to its scene bank, validation-selected
checkpoint, configuration and 32-layout test record. Original training,
validation and evaluation records are retained with the selected models.

Archived folders named exploratory_test contain the reported scenario,
component-ablation and coverage-target results, not additional unused sweeps.
Their records, checkpoint selections and summaries are retained for provenance.
Sources shared with the adopted training code and optional short-check functions
are also retained; no generated short-training outputs or development logs are
included.

These materials accompany the revised manuscript. Existing third-party
notices in the source remain applicable. Only load supplied trusted checkpoints.
