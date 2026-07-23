# BLCR

Reference implementation of **Balanced Learnability-Coverage Replay (BLCR)** for the paper *Balancing Learnability and Bellman-Error Coverage in Prioritized Experience Replay*.

BLCR is a conservative correction to reducible-loss prioritization (ReLo). ReLo remains the preferred ranking signal, while a compressed TD-error floor prevents difficult transitions from disappearing from replay:

$$
\widehat R_i=\operatorname{clip}\!\left(\frac{R_i}{m_R},0,c_{\max}\right),
\qquad
\widehat D_i=\operatorname{clip}\!\left(\frac{|\delta_i|}{m_D},0,c_{\max}\right),
$$

$$
p_i^{\mathrm{BLCR}}
=\operatorname{clip}\!\left(
\max\{\widehat R_i,\lambda\log(1+\widehat D_i)\},
p_{\min},p_{\max}\right).
$$

For a sampled transition, BLCR also applies a bounded learnability factor

$$
g_i=\operatorname{clip}\!\left(
1+\eta\log(1+\widehat R_i^{\,\ell}),1,g_{\max}\right)
$$

to the importance-corrected Bellman loss. The public method keys are `blcr` for discrete control and `blcr_sac` for continuous control.

## Repository Layout

- `fluid/agents/`: MinAtar and vector-observation DQN trainers.
- `fluid/replay/`: prioritized replay buffer and sum tree.
- `fluid/train_blcr_sac.py`: SAC, PER-SAC, one-step ReLo-SAC, and BLCR-SAC.
- `configs/`: paper configurations for MinAtar, ALE-RAM, and classic control.
- `scripts/`: complete experiment queues and result summarizers.
- `analysis/generate_figures.py`: learning-curve generation from completed logs.
- `docs/RESULTS_PROVENANCE.md`: provenance and integrity notes for reported results.

## Installation

The tested setup uses Python 3.11 and PyTorch 2.1 or newer.

```bash
conda env create -f environment.yml
conda activate BLCR
pip install -e .
AutoROM --accept-license
```

If Atari ROMs are already installed, the final command is unnecessary. MuJoCo is installed through the Gymnasium extra; no separate MuJoCo license key is required by current Gymnasium releases.

Run the lightweight MinAtar stub before launching long experiments:

```bash
python tests/smoke_minatar_stub.py
```

## Reproducing Experiments

All queue scripts accept `SEEDS`, `METHODS`, `MAX_PARALLEL`, `PYTHON_BIN`, and their domain-specific budget variables as environment overrides. They skip completed `result.pt` files by default.

MinAtar, five games, five seeds, two million frames:

```bash
bash scripts/run_minatar.sh
```

ALE-RAM, 30 games, ReLo versus BLCR, five seeds, two million frames:

```bash
bash scripts/run_ale_ram.sh
```

Gymnasium classic control, four tasks, four methods, three seeds, one million steps:

```bash
bash scripts/run_classic_control.sh
```

MuJoCo locomotion, four tasks, four SAC replay variants, three seeds, 300K steps:

```bash
bash scripts/run_mujoco_300k.sh
```

For a short functional check, override the budget and task set:

```bash
GAMES="breakout" SEEDS="0" METHODS="relo blcr" NUM_FRAMES=100000 \
  bash scripts/run_minatar.sh
```

Monitor any run with TensorBoard:

```bash
tensorboard --logdir logs_minatar_neurocomputing_2m
```

## Summaries and Figures

```bash
python scripts/summarize_discrete.py \
  --logdir logs_minatar_neurocomputing_2m

python scripts/summarize_ale_ram.py \
  --logdir logs_ale_ram_30games

python scripts/summarize_blcr_sac.py \
  --logdir logs_blcr_sac_locomotion_300k

python analysis/generate_figures.py
```

The discrete primary metric is the per-seed mean return over the final 100 episodes. The MuJoCo tables use the final deterministic evaluation at the stated training step, averaged across seeds.

## Continuous-Control Scope

The included `relo_sac` implementation uses realized one-step critic-loss reduction as its learnability proxy. This proxy is shared by `blcr_sac`, making their comparison controlled, but it is not a reproduction of the target-critic ReLo estimator used in the original ReLo SAC experiments. The paper reports this distinction explicitly.

## Research Integrity

Reported baseline values are read directly from completed run artifacts under a common aggregation rule. No baseline score is manually rescaled, lowered, or replaced. The ALE-RAM suite compares only ReLo and BLCR and therefore does not establish state-of-the-art ALE performance.
