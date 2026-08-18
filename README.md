# Metadata Grounding versus Prompt Engineering in Compute-Constrained Vision-Language Models

Reproduction code and released outputs for the paper *AI-Enabled Product Demonstration
Content for Direct Selling: Metadata Grounding versus Prompt Engineering in
Compute-Constrained Vision-Language Models*.

<!-- Add after archiving: [![DOI](https://zenodo.org/badge/DOI/XX.XXXX/zenodo.XXXXXXX.svg)](https://doi.org/XX.XXXX/zenodo.XXXXXXX) -->

## Summary

Three open-weight vision-language models (500M, 2B, and 4.2B parameters) generate product
demonstration scripts for 150 products from the Amazon Berkeley Objects catalogue, under
three prompt conditions, fully crossed and paired within product. 1,350 generations in
total, scored by rule-based comparison against the catalogue record. The 4.2B model,
Phi-3.5-Vision, was added during revision; the original two models (SmolVLM-500M and
Qwen2-VL-2B) reproduce their published values unchanged.

Supplying verified catalogue attributes in the prompt more than doubles attribute
coverage in every model (paired *d* = 1.19, 1.14, and 2.03). Rewriting the prompt with an
explicit output specification does not (paired *d* = 0.21, 0.18, and -0.05), with
non-overlapping confidence intervals between the two interventions. Grounded coverage
rises with model size while image-only coverage stays low and flat, and the per-product
benefit of grounding is idiosyncratic across models.

## Reproducing the results

The published tables and figures regenerate from the released outputs on CPU in about two
minutes. No GPU, no downloads, no API keys.

```bash
git clone https://github.com/<user>/<repo>.git && cd <repo>
pip install -r requirements.txt
jupyter notebook notebooks/abo_vlm_grounding.ipynb   # run all cells
```

Every reported value is a deterministic function of `results/scored_all.csv`, the full
per-generation record for all 1,350 outputs (including the generated text). The three-model
tables and figures are rebuilt from it.

`RUN_GENERATION = False` is the default. Set it to `True` on a machine with a CUDA GPU to
regenerate the two original models from scratch (roughly 1.5 hours on an NVIDIA T4).
The third model is regenerated with `run_third_model.py`, which reuses the identical
prompts and scorer; on the same free Colab T4 it adds the 450 Phi-3.5-Vision generations
and writes the merged `results/scored_all.csv` with an integrity check that re-scores the
two original models. The exact Colab procedure, including the library switch used for the
third model, is in `notebooks/phi35_third_model_reproduction.ipynb`.

The notebook also runs unmodified in Google Colab, including the free tier.

## Repository layout

```
notebooks/abo_vlm_grounding.ipynb   Pipeline for the two original models: sampling,
                                    generation, scoring, statistics, figures, tables
run_third_model.py                  Adds the third model (Phi-3.5-Vision) under the
                                    identical prompts, decoding and scorer; writes
                                    scored_all.csv with a two-model integrity check
notebooks/phi35_third_model_reproduction.ipynb  Exact Colab steps for the third-model run
environment_colab_phi.txt           Frozen pip environment of the third-model session
data/item_ids.csv                   The 150 sampled ABO item identifiers
results/gen_*.jsonl                 Raw generations, one JSON object per line, per model
                                    (qwen2vl-2b, smolvlm-500m, phi35-vision-4b)
results/scored_all.csv              All 1,350 per-generation records (three models)
results/scored.csv                  Original two-model subset (900 records)
results/manifest.csv                Sampled products with catalogue attributes
results/table1_main.csv             Paper Table 2 (main results, three models)
results/table2_effects.csv          Paper Table 3 (paired effect sizes, three models)
results/table3_threshold_sensitivity.csv   Threshold sensitivity
results/table4_sample_composition.csv      Sample composition
results/table5_scale_consistency.csv       Scale and cross-model consistency
results/figures/                    Paper figures, PNG and TIFF at 300 dpi
                                    (fig1 coverage, fig2 well-formedness, fig3 length,
                                    fig4 paired scatter, fig5 effect sizes, fig6 scale)
results/provenance.json             Config, versions, platform, record counts
audit/                              Blinded rating sheets, instructions and scoring
                                    script for the manual metric validation
audit/reviewer_response_analyses.py Supplementary revision analyses (category balance,
                                    length control, metric robustness); CPU, no GPU
```

## Mapping to the paper

| Paper artifact | Produced by |
|---|---|
| Table 2 (main results) | `table1_main.csv` |
| Table 3 (effect sizes) | `table2_effects.csv` |
| Threshold and composition tables | `table3_*.csv`, `table4_*.csv` |
| Scale and Cross-Model Consistency | `table5_scale_consistency.csv`, `fig6_scale` |
| Figures 1-6 | `results/figures/` |
| Replication on a third model | `run_third_model.py`, Phi rows in the tables |
| Category robustness, length control, metric robustness | `audit/reviewer_response_analyses.py` |
| Manual metric audit | `audit/` |

Every value in the paper is produced by a script from the released outputs. No number was
transcribed by hand.

## Reproducibility notes

Sampling is deterministic under seed `20260719`; an independent re-execution produced an
identical manifest. Decoding is greedy, so each generation is a deterministic function of
the model revision, library version and input.

Randomness is controlled by fixed seeds throughout, and there is more than one. Product
sampling and the category-balance resampling use seed `20260719`; the bootstrap confidence
intervals use fixed seeds (`0` for the coverage intervals, `1` for the paired effect-size
intervals); the scatter-plot jitter uses seed `0`. None of these affect the generated text,
which is greedy and therefore seed-independent.

Model revisions were not pinned in the published run, and the released records for that run
carry no revision identifier. Set `revision` in the model configuration to pin a commit
hash for exact reproduction. Repository contents can change over time, so a reader working
from repository names alone may obtain a later revision of a model.

Published run environment for the two original models: Python 3.12.13, transformers 4.51.3,
accelerate 1.6.0, huggingface-hub 0.36.2, pandas 2.2.2, numpy 2.0.2, single NVIDIA Tesla T4.
The third model was generated in a separate Colab session on the same T4 with a distinct
environment (transformers 4.43.0, accelerate 0.30.0, torch 2.11.0+cu128; frozen in
`environment_colab_phi.txt`);
because Phi-3.5-Vision exposes no separate system role, its system instruction is prepended
to the user turn, with information identical to that given to the other two models.

## Data licence and attribution

This study uses the Amazon Berkeley Objects dataset, released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Research use only;
commercial use is prohibited.

> Credit for the data, including all images and 3D models, is due to Matthieu Guillaumin
> (Amazon.com), Thomas Dideriksen (Amazon.com), Kenan Deng (Amazon.com), Himanshu Arora
> (Amazon.com), Jasmine Collins (UC Berkeley) and Jitendra Malik (UC Berkeley).

Collins, J., Goel, S., Deng, K., Luthra, A., Xu, L., Gundogdu, E., Zhang, X., Yago
Vicente, T. F., Dideriksen, T., Arora, H., Guillaumin, M., & Malik, J. (2022). *ABO:
Dataset and benchmarks for real-world 3D object understanding.* CVPR.
https://arxiv.org/abs/2110.06199

**This repository redistributes no ABO content.** It releases the sampled item identifiers
and the code that reconstructs the sample. `results/manifest.csv` contains catalogue
metadata for the 150 sampled products and is provided under the same CC BY-NC 4.0 terms.

## Models

- Marafioti, A., et al. (2025). *SmolVLM: Redefining small and efficient multimodal
  models.* https://arxiv.org/abs/2504.05299
- Wang, P., et al. (2024). *Qwen2-VL: Enhancing vision-language model perception of the
  world at any resolution.* https://arxiv.org/abs/2409.12191
- Microsoft (2024). *Phi-3.5-vision-instruct.* https://huggingface.co/microsoft/Phi-3.5-vision-instruct

## Licence

Code is released under the MIT Licence (see `LICENSE`). Data-derived artifacts inherit the
non-commercial restriction of the source dataset.

## Citation

See `CITATION.cff`. Update with the published reference once the article appears.
