# Progressive Candidate Correction for Olive Flounder

Official implementation of **Recognition of Visible Disease Symptoms in
*Paralichthys olivaceus* Using Progressive Candidate Correction**.

The network forms Part-structure information and four types of Visual evidence
from an RGB image, progressively corrects class-specific symptom candidates,
and interprets them through body, mouth, fin, and caudal-fin routes. This
repository contains the proposed method, training and evaluation commands,
candidate-correction comparisons, and semantic-segmentation baselines.

## Documentation

| Topic | Contents |
|---|---|
| [Method](docs/METHOD.pdf) | Network flow, notation, candidate correction, and Route-wise composition |
| [Supervision](docs/SUPERVISION.pdf) | Construction of Part-structure, Visual evidence, route, and boundary targets |
| [Loss functions](docs/LOSS_FUNCTIONS.pdf) | Loss equations, coefficients, masks, and stage multipliers |
| [Training](docs/TRAINING.pdf) | Six-stage schedule, trainable modules, optimizer, and augmentation |
| [Dataset](docs/DATASET.pdf) | Dataset layout, classes, split, preprocessing, and evaluation units |

The paper configuration is
[`configs/progressive_candidate_correction.yaml`](configs/progressive_candidate_correction.yaml),
and the network entry point is
[`ProgressiveCandidateCorrectionNetwork`](src/progressive_candidate_correction/models/network.py).

## Installation

Install a PyTorch build suitable for the target CPU or CUDA environment, then
install the package from the repository root:

```bash
python -m pip install -e ".[train,data,baselines,report]"
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Data

The study images are not distributed with this repository. Organize an
authorized local copy as follows:

```text
dataset-root/
  annotations/
    train_annotations.json
    val_annotations.json
  train/
    images/
    masks/
    part_masks/
  val/
    images/
    masks/
    part_masks/
```

The individual-wise split contains 1,169 training images from 640 fish and 292
validation images from 160 fish. Verify the file pairs and fish-identity split
before training:

```bash
python scripts/verify_dataset.py --data-root /path/to/dataset-root
```

## Training

The paper setting uses seed 45, 384 x 384 inputs, batch size 8, 120 epochs, a
RepViT-M1.1 encoder, and AdamW. The six-stage schedule is given in
[docs/TRAINING.pdf](docs/TRAINING.pdf).

```bash
python scripts/train.py \
  --config configs/progressive_candidate_correction.yaml \
  --data-root /path/to/dataset-root
```

CONCAT, SIGNED, and FULL use the configurations in
[`configs/correction`](configs/correction). U-Net and SegFormer comparisons use
[`configs/baselines`](configs/baselines).

## Evaluation

Evaluate the semantic output:

```bash
python scripts/evaluate.py \
  --config configs/progressive_candidate_correction.yaml \
  --checkpoint /path/to/model.pt \
  --data-root /path/to/dataset-root \
  --strict
```

Export the Initial symptom candidate response `R0` or Corrected symptom
candidate response `C` separately for the training and validation splits:

```bash
python scripts/evaluate.py \
  --config configs/progressive_candidate_correction.yaml \
  --checkpoint /path/to/model.pt \
  --data-root /path/to/dataset-root \
  --split train \
  --candidate-response C \
  --candidate-archive outputs/C_train_candidates.npz \
  --strict

python scripts/evaluate.py \
  --config configs/progressive_candidate_correction.yaml \
  --checkpoint /path/to/model.pt \
  --data-root /path/to/dataset-root \
  --split val \
  --candidate-response C \
  --candidate-archive outputs/C_validation_candidates.npz \
  --strict
```

Select class thresholds on the training responses and apply them unchanged to
the validation responses:

```bash
python scripts/evaluate_candidates.py \
  --train outputs/C_train_candidates.npz \
  --validation outputs/C_validation_candidates.npz \
  --protocol-config configs/evaluation/candidate_evaluation.yaml \
  --output outputs/candidate_metrics.json
```

The protocol uses class-specific 8-connected symptom regions, a 50% coverage
criterion, and the highest training threshold that reaches a 70% training
symptom-region capture rate. Repeat the export with `--candidate-response R0`
to compare the responses before and after correction.

## Paper notation

| Symbol | Model quantity |
|---|---|
| $F$ | Common feature |
| $S$ | Part-structure information: Fish Region, Part Map, Head-to-Tail Direction, and Zone Map |
| $E$ | Visual evidence: Redness, Shape, Lesion, and Unaffected Surface |
| $R_0$ | Initial symptom candidate response |
| $N$ | Signed-corrected candidate response |
| $A$ | Gated residual correction term |
| $C=N+A$ | Corrected symptom candidate response |
| $Z$ | Routed semantic logits |
| $Y$ | Auxiliary semantic mask |

$R_0$, $N$, and $C$ are independent foreground-class responses and may overlap
at a pixel. $Y$ is the mutually exclusive background-plus-symptom output.

## Citation and license

Please cite the accompanying manuscript. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff). Source code is licensed under Apache-2.0; see
[`NOTICE`](NOTICE) for the terms that apply to data and trained weights.
