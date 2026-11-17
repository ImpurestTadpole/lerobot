# Good-View JEPA

**Good-View JEPA** (`goodview_jepa`) is [VLA-JEPA](../vla_jepa/README.md) with a different world-model
target, motivated by Balestriero and LeCun, _Proving and Improving the Platonic Representation
Hypothesis with JEPAs_ ([arXiv:2610.39302](https://arxiv.org/abs/2610.39302)). VLA-JEPA's world
model predicts the next V-JEPA2 embeddings of the same camera views. Good-View JEPA instead
predicts what another camera sees of the same world state. Training and usage are covered in the
[docs page](../../../../docs/source/goodview_jepa.mdx); this README covers the ideas behind it.

---

## Fundamentals

### JEPAs

A Joint-Embedding Predictive Architecture (JEPA) learns an encoder by making the embeddings of two
related inputs (a _positive pair_) predictable from one another, in representation space rather
than pixel space. A regulariser stops the trivial solution where every input maps to the same
point. The paper uses the LeJEPA objective for this: an invariance term between the two views'
embeddings plus SIGReg, which pushes the embedding distribution towards an isotropic Gaussian
(weight λ = 0.02).

### Good views and bad views

The paper's central distinction is what the positive pair is:

- **Good views:** "two observations of world states that the world itself relates", such as two
  photographs, or two descriptions, of one scene.
- **Bad views:** "corruptions of a single observation", such as two crops of one photograph or two
  maskings of one sentence. Both views carry the same observation's nuisance (its lighting, its
  sensor noise, its wording), so the encoder can satisfy the objective by encoding that nuisance.

### What the paper proves

Under idealised assumptions (a finite world, injective renderings, population optima, and
minimisers unique up to rotation):

| Result        | Statement                                                                                                                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Theorem 1     | **Platonic Representation Theorem.** JEPAs trained separately on any number of modalities, with shared good-view positives, learn the same representation up to one rotation.                           |
| Proposition 2 | k paired examples spanning the embedding space identify that rotation (orthogonal Procrustes); unpaired statistics cannot.                                                                              |
| Theorem 2     | When the views are not shared exactly, misalignment is bounded by the mismatch between the two positive-pair kernels, divided by an eigengap.                                                           |
| Theorem 3     | **Nuisance costs alignment.** Every embedding direction spent on a modality's own nuisance is lost for alignment (assumes nuisance drawn independently of the state).                                   |
| Proposition 3 | Good-view JEPAs learn a function of the world state alone; bad-view JEPAs can spend embedding directions on nuisance, with an alignment error that grows with them.                                     |
| Proposition 4 | Next-token prediction is a JEPA whose positives are a prefix and its continuation in one document: a bad-view JEPA. The paper uses this to explain why language models align with vision only at scale. |

Empirically, good views reached the same cross-modal alignment as next-token prediction with 42x
fewer samples, and a 109M-parameter text JEPA aligned with DINOv2 and CLIP image encoders better
than every language model the authors evaluated (up to 72B parameters). The paper's conclusion is
that the choice of positive views, not scale alone, drives shared ("platonic") representations.

The paper measures alignment with mutual k-nearest neighbours (no fitting), with retrieval after one
fitted rotation, and with linear CKA. It also reports that without gradient clipping, a few of its
runs collapsed during training, and recommends clipping.

---

## From the paper to robot cameras

On a robot, two cameras recording the same moment are two observations of one world state, so they
form a good-view pair. Two frames from one camera share that camera's viewpoint, exposure, lens and
noise, which is closer to the paper's bad views. That last point is an interpretation: the paper
defines bad views as corruptions of a single observation, not as consecutive frames.

Good-View JEPA applies this to VLA-JEPA's world-model target through `cross_view_mode`:

| `cross_view_mode`  | Predictor input        | Target                            | Pair type                |
| ------------------ | ---------------------- | --------------------------------- | ------------------------ |
| `none`             | each camera at steps t | the same camera at t+1 (VLA-JEPA) | same camera              |
| `same_time`        | camera `src` at step t | camera `dst` at step t            | good view                |
| `future` (default) | camera `src` at step t | camera `dst` at step t+1          | good view, plus dynamics |

`same_time` gives the action tokens nothing to predict, since no time passes between input and
target, so it weakens the latent-action signal VLA-JEPA is built on. `future` keeps that signal and
is the default.

### Where the theory stops

This policy is an experiment the paper motivates, not one it proves. In particular:

- **The encoder is frozen.** The V-JEPA2 encoder in VLA-JEPA is not trained, so changing the target
  changes what the world-model predictor and Qwen's action tokens learn, not the encoder's
  representation the theorems are about.
- **Action-conditioned predictors are not covered.** The paper states its theory does not cover JEPAs
  with a learned predictor, such as action-conditioned world models.
- **Partial views.** A wrist camera often cannot see what the other camera sees. The paper's
  Theorem 3 assumes each observation determines the state; partial views fall under the weaker
  Theorem 2 bound, so some misalignment remains.
- **Shared nuisance.** Theorem 3 assumes nuisance independent of the state. Lighting changes, shadows
  or motion blur that hit every camera at once are shared nuisance and can still be learned.

### Practical conditions

For camera pairs to behave as good views:

- **Synchronise the cameras.** Frames that are not from the same instant are not the same world
  state. Use hardware timestamps where possible and aim for under one frame of offset.
- **Keep nuisance independent per camera.** Separate exposure per camera helps; the photometric
  augmentation in `examples/goodview_jepa` is drawn independently per view.
- **Check camera overlap.** The world model uses the first two cameras in `image_features` order;
  choose `cross_view_source` and `cross_view_dest` so the two cameras see the task.

---

## Architecture

Identical to VLA-JEPA; only the world-model target, the clip frame spacing (`video_frame_stride`)
and optional training-time camera blanking (`camera_blank_prob`) differ.

| Component               | Module                            | Role                                                                   |
| ----------------------- | --------------------------------- | ---------------------------------------------------------------------- |
| **Qwen3-VL backbone**   | `Qwen3VLInterface`                | Fuses images and the instruction into context and action tokens        |
| **DiT action head**     | `VLAJEPAActionHead`               | Flow-matching denoising of the action chunk                            |
| **V-JEPA2 world model** | `ActionConditionedVideoPredictor` | Predicts the target camera's frozen V-JEPA2 embeddings (training only) |
| **Cross-view target**   | `GoodViewJEPAModel`               | Selects the predictor's input and target cameras                       |

At inference only the Qwen backbone and action head run, exactly as in VLA-JEPA. VLA-JEPA
checkpoints load unchanged after `convert_from_vla_jepa` relabels their config.

---

## Citation

The idea:

```bibtex
@misc{balestriero2026platonic,
  title         = {Proving and Improving the Platonic Representation Hypothesis with JEPAs},
  author        = {Randall Balestriero and Yann LeCun},
  year          = {2026},
  eprint        = {2610.39302},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.39302},
}
```

The model it builds on:

```bibtex
@misc{sun2026vlajepaenhancingvisionlanguageactionmodel,
  title         = {VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model},
  author        = {Jingwen Sun and Wenyao Zhang and Zekun Qi and Shaojie Ren and Zezhi Liu and Hanxin Zhu and Guangzhong Sun and Xin Jin and Zhibo Chen},
  year          = {2026},
  eprint        = {2602.10098},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2602.10098},
}
```
