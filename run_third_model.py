# =============================================================================
# run_third_model.py  —  Add a THIRD VLM to the ABO grounding study
#
# Reuses the EXACT prompts, generation settings, and scoring from
# notebooks/abo_vlm_grounding.ipynb so the new model's numbers are directly
# comparable to SmolVLM-500M and Qwen2-VL-2B.
#
# HOW TO RUN (Google Colab, GPU runtime — T4 is enough):
#   1. Runtime > Change runtime type > GPU (T4).
#   2. Clone your repo so results/manifest.csv is present:
#        !git clone https://github.com/LoveleenGaur/abo-vlm-grounding-repo.git
#        %cd abo-vlm-grounding-repo
#      (or upload manifest.csv and set MANIFEST_PATH below)
#   3. !python run_third_model.py         # or paste this whole file into a cell
#   4. Send me the file it writes:  results/scored_all.csv
#
# It downloads only the 150 product images it needs (small ABO variant, ~a few MB
# each), runs the third model over the same 150 x 3 conditions, scores every
# output with the identical rule-based scorer, and writes:
#   results/gen_<key>.jsonl      raw generations for the third model
#   results/scored_all.csv       ALL THREE models, one schema  <-- send me this
# =============================================================================

import gzip, json, os, re, time, platform
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 0. CHOOSE THE THIRD MODEL
#    "phi35"    -> microsoft/Phi-3.5-vision-instruct  (~4.2B; distinct architecture
#                  AND a new parameter scale -> answers R2's "architectures AND
#                  scales" on both axes).  RECOMMENDED.
#    "llava_ov" -> llava-hf/llava-onevision-qwen2-0.5b-ov-hf  (0.5B; distinct
#                  LLaVA-OneVision design; native transformers, no remote code;
#                  lightest option if GPU/RAM is tight).
# ----------------------------------------------------------------------------
MODEL_CHOICE = "phi35"          # <- change to "llava_ov" if you prefer

MODEL_TABLE = {
    "phi35":    {"key": "phi35-vision-4b",  "repo": "microsoft/Phi-3.5-vision-instruct",
                 "label": "Phi-3.5-Vision (4.2B)"},
    "llava_ov": {"key": "llava-ov-500m",    "repo": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
                 "label": "LLaVA-OneVision (0.5B)"},
}
SPEC = MODEL_TABLE[MODEL_CHOICE]

# Identical to the notebook CONFIG
CONFIG = {"n_products": 150, "seed": 20260719, "max_new_tokens": 150,
          "usable_min_trigram": 0.6,
          "conditions": ["image_meta", "image_only", "image_only_scaffold"]}
S3 = "https://amazon-berkeley-objects.s3.amazonaws.com"

MANIFEST_PATH = os.environ.get("MANIFEST_PATH", "")   # optional override

# ----------------------------------------------------------------------------
# 1. Locate the shipped manifest (the SAME 150 products; do NOT resample)
# ----------------------------------------------------------------------------
def find_manifest():
    if MANIFEST_PATH:
        return Path(MANIFEST_PATH)
    for c in ["results/manifest.csv", "manifest.csv",
              "abo-vlm-grounding-repo/results/manifest.csv",
              "/content/abo-vlm-grounding-repo/results/manifest.csv"]:
        if Path(c).exists():
            return Path(c)
    raise SystemExit(
        "manifest.csv not found. Clone your repo and run from its root, or set "
        "MANIFEST_PATH=/path/to/manifest.csv")

MANIFEST = find_manifest()
ROOT = MANIFEST.parent
WORK = Path("./.cache"); (WORK / "images").mkdir(parents=True, exist_ok=True)
sample = pd.read_csv(MANIFEST)
assert len(sample) == 150, f"expected 150 products, manifest has {len(sample)}"
print(f"manifest: {len(sample)} products, {sample.product_type.nunique()} types  ->  {MANIFEST}")
print(f"third model: {SPEC['label']}  [{SPEC['repo']}]")

# ----------------------------------------------------------------------------
# 2. Download the 150 images (identical logic to notebook Part 1)
# ----------------------------------------------------------------------------
import requests
def fetch(url, dest, retries=3):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.rename(dest)
            return dest
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    return dest

print("resolving image paths ...")
idx = fetch(f"{S3}/images/metadata/images.csv.gz", WORK / "images.csv.gz")
id_to_path = dict(pd.read_csv(idx)[["image_id", "path"]].values)
image_files = {}
for _, row in sample.iterrows():
    rel = id_to_path.get(row["main_image_id"])
    if rel is None:
        continue
    dest = WORK / "images" / f"{row['item_id']}.jpg"
    try:
        fetch(f"{S3}/images/small/{rel}", dest)
        image_files[row["item_id"]] = dest
    except Exception as e:
        print(f"  skip {row['item_id']}: {e}")
sample = sample[sample.item_id.isin(image_files)].reset_index(drop=True)
print(f"{len(sample)} products with images")

# ----------------------------------------------------------------------------
# 3. Prompts — COPIED VERBATIM from the notebook (Part 2)
# ----------------------------------------------------------------------------
SYSTEM = ("You are helping a direct-selling distributor demonstrate a product to a "
          "customer. Describe only what you can verify. Do not invent specifications.")
TASK = ("Write a short spoken product demonstration script (about 80 words). "
        "Mention what the product is, its key visible features, and one practical benefit.")
SCAFFOLD = ("Write a spoken product demonstration script of 60-100 words.\n"
            "Structure it in exactly three sentences:\n"
            "1. What the product is.\n"
            "2. Two features you can see in the image.\n"
            "3. One practical benefit for the customer.\n"
            "Write the script only. Do not write a label or a single word.")

def build_prompt(row, condition):
    if condition == "image_only":
        return TASK
    if condition == "image_only_scaffold":
        return SCAFFOLD
    facts = [f"Product type: {row['product_type']}"]
    for key in ["item_name", "brand", "color", "material", "style"]:
        v = row.get(key)
        if pd.notna(v) and v:
            facts.append(f"{key.replace('_',' ').title()}: {v}")
    return f"{TASK}\n\nVerified product information:\n" + "\n".join(facts)

# ----------------------------------------------------------------------------
# 4. Model adapters. Same greedy decoding (do_sample=False), same
#    max_new_tokens (150), same 512px thumbnail. SYSTEM is prepended to the user
#    turn (these models have no separate system role); the information content is
#    identical to how SmolVLM/Qwen received it.
# ----------------------------------------------------------------------------
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print("device:", DEVICE)
if DEVICE == "cpu":
    print("WARNING: no GPU detected — generation will be extremely slow. "
          "Switch Colab to a GPU runtime.")

def load_model():
    if MODEL_CHOICE == "phi35":
        from transformers import AutoModelForCausalLM, AutoProcessor
        proc = AutoProcessor.from_pretrained(SPEC["repo"], trust_remote_code=True, num_crops=4)
        model = AutoModelForCausalLM.from_pretrained(
            SPEC["repo"], torch_dtype=DTYPE, device_map=DEVICE,
            trust_remote_code=True, _attn_implementation="eager").eval()
        return model, proc
    else:  # llava_ov — native, no remote code
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        proc = AutoProcessor.from_pretrained(SPEC["repo"])
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            SPEC["repo"], torch_dtype=DTYPE, device_map=DEVICE).eval()
        return model, proc

def generate(model, proc, img, prompt_text):
    user_text = f"{SYSTEM}\n\n{prompt_text}"
    if MODEL_CHOICE == "phi35":
        messages = [{"role": "user", "content": f"<|image_1|>\n{user_text}"}]
        prompt = proc.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(prompt, [img], return_tensors="pt").to(DEVICE)
        in_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"],
                                 do_sample=False,
                                 eos_token_id=proc.tokenizer.eos_token_id)
        trimmed = ids[:, in_len:]
        return proc.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0].strip()
    else:  # llava_ov
        conv = [{"role": "user", "content": [{"type": "text", "text": user_text},
                                             {"type": "image"}]}]
        prompt = proc.apply_chat_template(conv, add_generation_prompt=True)
        inputs = proc(images=img, text=prompt, return_tensors="pt").to(DEVICE, DTYPE)
        in_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"],
                                 do_sample=False)
        trimmed = ids[:, in_len:]
        return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

# ----------------------------------------------------------------------------
# 5. Generate (resumable — safe to re-run if the session drops)
# ----------------------------------------------------------------------------
def done_keys(path):
    keys = set()
    if Path(path).exists():
        for line in open(path):
            try:
                r = json.loads(line); keys.add((r["item_id"], r["condition"]))
            except Exception:
                pass
    return keys

out_path = ROOT / f"gen_{SPEC['key']}.jsonl"
completed = done_keys(out_path)
todo = [(r, c) for _, r in sample.iterrows()
        for c in CONFIG["conditions"] if (r["item_id"], c) not in completed]
print(f"to generate: {len(todo)} of {len(sample)*3}")

if todo:
    model, proc = load_model()
    gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    with open(out_path, "a") as sink:
        for n, (row, cond) in enumerate(todo, 1):
            img = Image.open(image_files[row["item_id"]]).convert("RGB")
            img.thumbnail((512, 512))
            t0 = time.time()
            try:
                text = generate(model, proc, img, build_prompt(row, cond))
            except Exception as e:
                text = ""; print(f"  gen error {row['item_id']}/{cond}: {e}")
            sink.write(json.dumps({
                "item_id": row["item_id"], "model": SPEC["key"], "condition": cond,
                "output": text, "seconds": round(time.time() - t0, 2),
                "gpu": gpu_name, "revision": None}) + "\n")
            sink.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")
    print("generation done ->", out_path.name)

# ----------------------------------------------------------------------------
# 6. Scoring — COPIED VERBATIM from the notebook (Part 3)
# ----------------------------------------------------------------------------
COLORS = {"black","white","grey","gray","silver","gold","beige","brown","red","blue",
          "navy","green","yellow","orange","pink","purple","ivory","cream"}
MATERIALS = {"wood","wooden","metal","steel","aluminum","aluminium","plastic","glass",
             "leather","cotton","linen","silk","ceramic","bamboo","marble","rubber",
             "polyester","velvet","brass","copper"}
PROMPT_MARKERS = ["two features","one practical benefit","60-100 words","product:",
                  "write a spoken","you can see in the image"]
SCORED_ATTRS = ["brand", "color", "material", "product_type"]

def norm(t):
    return re.sub(r"[^a-z0-9 ]", " ", str(t).lower())

def rep_ratio(text, n=3):
    w = str(text).lower().split()
    if len(w) < n + 1:
        return 0.0
    g = [tuple(w[i:i+n]) for i in range(len(w)-n+1)]
    return len(set(g)) / len(g)

def score_record(rec, gt):
    text = norm(rec["output"]); tokens = set(text.split())
    available = covered = 0
    for a in SCORED_ATTRS:
        v = gt.get(a)
        if pd.isna(v) or not v:
            continue
        available += 1
        if any(w in text for w in norm(v).split() if len(w) > 2):
            covered += 1
    supported = set(norm(f"{gt.get('color','')} {gt.get('material','')} "
                         f"{gt.get('item_name','')}").split())
    return {"coverage": covered / available if available else np.nan,
            "n_attrs": available,
            "unsupported_terms": len((tokens & (COLORS | MATERIALS)) - supported),
            "words": len(text.split()),
            "echo": any(m in str(rec["output"]).lower() for m in PROMPT_MARKERS),
            "reps": rep_ratio(rec["output"])}

gt_map = pd.read_csv(MANIFEST).set_index("item_id").to_dict("index")

def score_jsonl(path):
    rows = []
    for line in open(path):
        rec = json.loads(line); gt = gt_map.get(rec["item_id"])
        if gt:
            rows.append({**rec, **score_record(rec, gt)})
    d = pd.DataFrame(rows)
    d["usable"] = ~(d.echo | (d.reps < CONFIG["usable_min_trigram"]))
    d["unsup_any"] = d.unsupported_terms > 0
    return d

new = score_jsonl(out_path)
SCHEMA = ["item_id","model","condition","output","seconds","gpu","coverage",
          "n_attrs","unsupported_terms","words","echo","reps","usable","unsup_any"]
for c in SCHEMA:
    if c not in new.columns:
        new[c] = np.nan
new = new[SCHEMA]

# ----------------------------------------------------------------------------
# 7. Integrity check: re-score the TWO shipped models from their jsonl and
#    confirm they reproduce results/scored.csv exactly. If this passes, the
#    third model's scores are guaranteed to be on the same footing.
# ----------------------------------------------------------------------------
shipped = ROOT / "scored.csv"
if shipped.exists():
    old = pd.read_csv(shipped)
    ok = True
    for key in ["qwen2vl-2b", "smolvlm-500m"]:
        j = ROOT / f"gen_{key}.jsonl"
        if j.exists():
            re_scored = score_jsonl(j)
            a = re_scored.sort_values(["item_id","condition"])["coverage"].round(6).values
            b = old[old.model==key].sort_values(["item_id","condition"])["coverage"].round(6).values
            match = len(a)==len(b) and np.allclose(np.nan_to_num(a,nan=-1), np.nan_to_num(b,nan=-1))
            print(f"integrity {key}: {'MATCH' if match else 'MISMATCH'}")
            ok = ok and match
    combined = pd.concat([old[SCHEMA], new], ignore_index=True)
else:
    print("note: results/scored.csv not found; writing third-model rows only")
    combined = new

out_csv = ROOT / "scored_all.csv"
combined.to_csv(out_csv, index=False)

# ----------------------------------------------------------------------------
# 8. Quick preview of the grounding effect for the third model (paired,
#    usable outputs). Report it whatever it shows.
# ----------------------------------------------------------------------------
u = new[new.usable]
piv = u.pivot_table(index="item_id", columns="condition", values="coverage")
if {"image_meta","image_only"}.issubset(piv.columns):
    pair = piv[["image_meta","image_only"]].dropna()
    d = pair["image_meta"] - pair["image_only"]
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) else float("nan")
    print("\n=== THIRD MODEL preview (usable, paired) ===")
    print(f"model: {SPEC['label']}  n_pairs={len(pair)}")
    print(f"coverage image_only={pair['image_only'].mean():.3f}  "
          f"image+metadata={pair['image_meta'].mean():.3f}  "
          f"grounding delta={d.mean():+.3f}  paired d={dz:.2f}")
if {"image_only_scaffold","image_only"}.issubset(piv.columns):
    pair2 = piv[["image_only_scaffold","image_only"]].dropna()
    d2 = pair2["image_only_scaffold"] - pair2["image_only"]
    dz2 = d2.mean()/d2.std(ddof=1) if d2.std(ddof=1) else float("nan")
    print(f"prompting (scaffold) delta={d2.mean():+.3f}  paired d={dz2:.2f}")

print(f"\nrows written: {len(combined)}  ({new.model.nunique()} new-model rows: {len(new)})")
print("versions:", platform.python_version(),
      "| transformers", __import__('transformers').__version__,
      "| torch", torch.__version__)
print(f"\n>>> SEND ME THIS FILE:  {out_csv.resolve()}")
