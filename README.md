# Metadata Grounding versus Prompt Engineering in Compute-Constrained Vision-Language Models

Reproduction code and released outputs for the paper *AI-Enabled Product Demonstration
Content for Direct Selling: Metadata Grounding versus Prompt Engineering in
Compute-Constrained Vision-Language Models*.

<!-- Add after archiving: [![DOI](https://zenodo.org/badge/DOI/XX.XXXX/zenodo.XXXXXXX.svg)](https://doi.org/XX.XXXX/zenodo.XXXXXXX) -->

## Summary

Two open-weight vision-language models (500M and 2B parameters) generate product
demonstration scripts for 150 products from the Amazon Berkeley Objects catalogue,
under three prompt conditions, fully crossed and paired within product. 900 generations
in total, scored by rule-based comparison against the catalogue record.

Supplying verified catalogue attributes in the prompt more than doubles attribute
coverage (paired *d* = 1.19 and 1.14). Rewriting the prompt with an explicit output
specification does not (paired *d* = 0.21 and 0.18), with non-overlapping confidence
intervals between the two interventions.

## Reproducing the results

The published tables and figures regenerate from the released outputs on CPU in about
two minutes. No GPU, no downloads, no API keys.

```bash
git clone https://github.com/<user>/<repo>.git && cd <repo>
pip install -r requirements.txt
jupyter notebook notebooks/abo_vlm_grounding.ipynb   # run all cells
```

`RUN_GENERATION = False` is the default. Set it to `True` on a machine with a CUDA GPU
to regenerate all 900 outputs from scratch; that takes roughly 1.5 hours on an NVIDIA
T4 and re-downloads the sampled catalogue images.

The notebook also runs unmodified in Google Colab, including the free tier.

## Repository layout

```
notebooks/abo_vlm_grounding.ipynb   Complete pipeline: sampling, generation,
                                    scoring, validation, statistics, figures, tables
data/item_ids.csv                   The 150 sampled ABO item identifiers
results/gen_*.jsonl                 All 900 raw generations, one JSON object per line
results/manifest.csv                Sampled products with catalogue attributes
results/scored.csv                  Per-generation metric values
results/table1_main.csv             Paper Table 1
results/table2_effects.csv          Paper Table 2
results/table3_*.csv                Paper Table 3
results/table4_*.csv                Paper Table 4
results/figures/                    Paper Figures 1-5, PNG and TIFF at 300 dpi
results/provenance.json             Config, versions, platform, record counts
audit/                              Blinded rating sheets, instructions and scoring
                                    script for the manual metric validation
audit/reviewer_response_analyses.py Supplementary revision analyses (category balance,
                                    length control, metric robustness); CPU, no GPU
```

## Mapping to the paper

| Paper artifact | Produced by |
|---|---|
| Table 1 | Notebook Part 7 |
| Table 2 | Notebook Part 5 |
| Tables 3-4 | Notebook Part 7 |
| Figures 1-5 | Notebook Part 6 |
| Section 6.6 category robustness | Notebook Part 4.2 |
| Section 8 metric audit | Notebook Part 4.3 and `audit/` |
| Sections 6.9-6.11 revision robustness | `audit/reviewer_response_analyses.py` |

Every value in the paper is produced by a script from `results/gen_*.jsonl`. No number
was transcribed by hand.

## Reproducibility notes

Sampling is deterministic under seed `20260719`; an independent re-execution produced an
identical manifest. Decoding is greedy, so each generation is a deterministic function
of the model revision, library version and input.

Model revisions were not pinned in the published run, and the released records for that
run carry no revision identifier. Set `revision` in `CONFIG["models"]` to pin a commit
hash for exact reproduction. Repository contents can change over time, so a reader
working from repository names alone may obtain a later revision of either model.

Published run environment: Python 3.12.13, transformers 4.51.3, accelerate 1.6.0,
huggingface-hub 0.36.2, pandas 2.2.2, numpy 2.0.2, single NVIDIA Tesla T4.

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

**This repository redistributes no ABO content.** It releases the sampled item
identifiers and the code that reconstructs the sample. `results/manifest.csv` contains
catalogue metadata for the 150 sampled products and is provided under the same
CC BY-NC 4.0 terms.

## Models

- Marafioti, A., et al. (2025). *SmolVLM: Redefining small and efficient multimodal
  models.* https://arxiv.org/abs/2504.05299
- Wang, P., et al. (2024). *Qwen2-VL: Enhancing vision-language model perception of the
  world at any resolution.* https://arxiv.org/abs/2409.12191

## Licence

Code is released under the MIT Licence (see `LICENSE`). Data-derived artifacts inherit
the non-commercial restriction of the source dataset.

## Citation

See `CITATION.cff`. Update with the published reference once the article appears.
