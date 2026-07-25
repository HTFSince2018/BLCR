# BLCR

Official reference implementation of **Balanced Learnability-Coverage Replay
(BLCR)** for *Balancing Learnability and Bellman-Error Coverage in Prioritized
Experience Replay*.

BLCR extends reducible-loss prioritization (ReLo) with a conservative coverage
correction. ReLo favors transitions that can produce immediate learning
progress, but it may repeatedly suppress transitions whose Bellman error
remains high. BLCR retains the ReLo ranking signal while adding a bounded
TD-error floor, so difficult yet informative experience cannot disappear from
replay.

## Method

For transition \(i\), let \(R_i\) be its reducible-loss signal and
\(D_i=|\delta_i|\) its absolute TD error. The signals are normalized by running
scales \(m_R\) and \(m_D\):

$$
\widehat R_i =
\operatorname{clip}\left(\frac{R_i}{m_R},0,c_{\max}\right),
\qquad
\widehat D_i =
\operatorname{clip}\left(\frac{D_i}{m_D},0,c_{\max}\right).
$$

BLCR assigns the replay priority

$$
p_i^{\mathrm{BLCR}} =
\operatorname{clip}\left(
\max\left\{\widehat R_i,\lambda\log(1+\widehat D_i)\right\},
p_{\min},p_{\max}
\right).
$$

The maximum preserves ReLo whenever its learnability estimate is sufficiently
large. The logarithmic floor restores coverage when the ReLo signal is small,
while compressing extreme TD errors. A bounded factor can also emphasize
learnable samples in the importance-corrected Bellman objective:

$$
\mathcal L_{\mathrm{BLCR}} =
\frac{1}{B}\sum_{i=1}^{B} w_i g_i\ell_i,
\qquad
g_i =
\operatorname{clip}\left(
1+\eta\log(1+\widehat R_i),1,g_{\max}
\right),
$$

where \(w_i\) is the importance-sampling correction and \(\ell_i\) is the
per-transition Bellman loss. The SAC adapter uses the normalized critic TD
residual in the bounded loss factor and shares the same learnability-coverage
priority structure.

## Supported Experiments

| Domain | Entry point | Methods |
| --- | --- | --- |
| MinAtar | `BLCR.train_minatar` | DQN, PER, ReLo, BLCR |
| ALE with RAM observations | `BLCR.train_external_discrete` | DQN, PER, ReLo, BLCR |
| Gymnasium classic control | `BLCR.train_external_discrete` | DQN, PER, ReLo, BLCR |
| MuJoCo locomotion | `BLCR.train_blcr_sac` | SAC, PER-SAC, ReLo-SAC, BLCR-SAC |

The public method keys are `blcr` for discrete control and `blcr_sac` for
continuous control.

## Repository Layout

```text
BLCR/
|-- BLCR/
|   |-- agents/       # DQN trainers
|   |-- envs/         # Gymnasium and ALE environment adapters
|   |-- models/       # Q-network definitions
|   |-- replay/       # Replay buffer and sum-tree implementation
|   |-- train_blcr_sac.py
|   |-- train_external_discrete.py
|   `-- train_minatar.py
|-- configs/
|   |-- external/
|   `-- minatar/
|-- environment.yml
|-- requirements.txt
`-- pyproject.toml
```

## Installation

The tested environment uses Python 3.11 and PyTorch 2.1 or newer.

```bash
conda env create -f environment.yml
conda activate BLCR
pip install -e .
AutoROM --accept-license
```

`AutoROM` is only required when Atari ROMs have not already been installed.
Gymnasium supplies the current MuJoCo bindings; a separate MuJoCo license key
is not required.

Alternatively, install the Python dependencies directly:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Run BLCR on MinAtar Breakout:

```bash
python -m BLCR.train_minatar \
  --config configs/minatar/blcr.yaml \
  --game breakout \
  --seed 0
```

Run two million frames on ALE Breakout with RAM observations:

```bash
python -m BLCR.train_external_discrete \
  --config configs/external/ale_ram_blcr.yaml \
  --algo blcr \
  --seed 0 \
  --override training.num_frames=2000000
```

Run BLCR on Gymnasium CartPole:

```bash
python -m BLCR.train_external_discrete \
  --config configs/external/gym_classic_control.yaml \
  --env-name cartpole_v1 \
  --algo blcr \
  --seed 0 \
  --override env.id=CartPole-v1
```

Run BLCR-SAC on Hopper:

```bash
python -m BLCR.train_blcr_sac \
  --env Hopper-v5 \
  --method blcr_sac \
  --seed 0 \
  --total-steps 300000
```

Configuration values can be changed without editing YAML files by repeating
`--override key=value`. For example:

```bash
python -m BLCR.train_minatar \
  --config configs/minatar/blcr.yaml \
  --game asterix \
  --seed 1 \
  --override training.num_frames=1000000 \
  --override system.device=cuda
```

## Outputs

Discrete-control runs write `metrics.jsonl`, TensorBoard events, and `result.pt`
under the configured log directory. Continuous-control runs store the same
core artifacts under:

```text
<logdir>/<environment>/<method>/seed<seed>/
```

TensorBoard can monitor a run directory while training:

```bash
tensorboard --logdir logs
```

## Experimental Scope

The discrete implementation contains the BLCR comparison paths used for
MinAtar, ALE-RAM, and classic-control experiments. The continuous-control
implementation compares replay variants within a common SAC training pipeline.
Its `relo_sac` path estimates learnability from realized one-step critic-loss
reduction; this controlled proxy is not presented as an exact reproduction of
every estimator used in the original ReLo continuous-control experiments.

Reported results should be aggregated from completed run artifacts with the
same metric and seed policy for every method. No baseline score should be
manually rescaled, lowered, or selectively replaced.
