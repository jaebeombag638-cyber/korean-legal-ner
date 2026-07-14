import json
import glob
from collections import Counter, defaultdict
import statistics

from transformers import AutoTokenizer

BASE_TRAIN = "data/12.판결서 익명처리 데이터/3.개방데이터/2.데이터(NIA)/Training/02.라벨링데이터"
BASE_VAL = "data/12.판결서 익명처리 데이터/3.개방데이터/2.데이터(NIA)/Validation/02.라벨링데이터"

files = glob.glob(f"{BASE_TRAIN}/**/*.json", recursive=True) + glob.glob(f"{BASE_VAL}/**/*.json", recursive=True)

rule_counter = Counter()
method_counter = Counter()
caseclass_counter = Counter()
rule_example_text = defaultdict(list)

total_spans = 0
mismatch_placeholder = 0
mismatch_drift = 0
match_ok = 0

section_lengths_chars = []
section_lengths_tokens = []

annotation_counts_per_doc = []

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

for i, fp in enumerate(files):
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)

    caseclass_counter[data["info"].get("caseClass", "UNKNOWN")] += 1
    annotation_counts_per_doc.append(data["info"].get("annotation_count", len(data.get("annotations", []))))

    sec_map = {s["section_id"]: s["text"] for s in data["sections"]}
    for s in data["sections"]:
        section_lengths_chars.append(s["char_count"])

    for ann in data.get("annotations", []):
        rule = ann.get("rules_triggered", "UNKNOWN")
        method = ann.get("method", "UNKNOWN")
        rule_counter[rule] += 1
        method_counter[method] += 1

        if len(rule_example_text[rule]) < 5:
            rule_example_text[rule].append(ann.get("original_text", ""))

        for sid in ann["section_id"]:
            if sid not in sec_map:
                continue
            text = sec_map[sid]
            for span in ann["span"]:
                total_spans += 1
                extracted = text[span["start"]:span["end"]]
                orig = ann["original_text"]
                if extracted == orig:
                    match_ok += 1
                elif "#" in orig or "#" in extracted:
                    mismatch_placeholder += 1
                else:
                    mismatch_drift += 1

    if i % 2000 == 0:
        print(f"progress: {i}/{len(files)}")

# 512 token truncation check: tokenize a sample of long sections in batches (전체 대상, 배치 인코딩)
print("tokenizing all sections for length check (batched)...")
all_section_texts = []
for fp in files:
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    for s in data["sections"]:
        all_section_texts.append(s["text"])

BATCH = 1000
over_512 = 0
for i in range(0, len(all_section_texts), BATCH):
    batch = all_section_texts[i:i+BATCH]
    enc = tokenizer(batch, truncation=False, add_special_tokens=True)
    for ids in enc["input_ids"]:
        section_lengths_tokens.append(len(ids))
        if len(ids) > 512:
            over_512 += 1
    if i % 5000 == 0:
        print(f"tokenizing progress: {i}/{len(all_section_texts)}")

results = {
    "total_files": len(files),
    "rule_counter": dict(rule_counter.most_common()),
    "method_counter": dict(method_counter.most_common()),
    "caseclass_counter": dict(caseclass_counter.most_common()),
    "rule_example_text": {k: v for k, v in rule_example_text.items()},
    "span_check": {
        "total_spans": total_spans,
        "match_ok": match_ok,
        "mismatch_placeholder": mismatch_placeholder,
        "mismatch_drift": mismatch_drift,
        "match_ok_pct": round(match_ok / total_spans * 100, 2),
        "mismatch_placeholder_pct": round(mismatch_placeholder / total_spans * 100, 2),
        "mismatch_drift_pct": round(mismatch_drift / total_spans * 100, 2),
    },
    "section_length": {
        "total_sections": len(section_lengths_tokens),
        "over_512_tokens": over_512,
        "over_512_pct": round(over_512 / len(section_lengths_tokens) * 100, 2),
        "mean_tokens": round(statistics.mean(section_lengths_tokens), 1),
        "median_tokens": statistics.median(section_lengths_tokens),
        "max_tokens": max(section_lengths_tokens),
    },
    "annotation_count_per_doc": {
        "mean": round(statistics.mean(annotation_counts_per_doc), 2),
        "median": statistics.median(annotation_counts_per_doc),
        "max": max(annotation_counts_per_doc),
        "min": min(annotation_counts_per_doc),
    },
}

with open("scripts/_eda_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("DONE")
print(json.dumps(results["rule_counter"], ensure_ascii=False, indent=2))
print(json.dumps(results["span_check"], ensure_ascii=False, indent=2))
print(json.dumps(results["section_length"], ensure_ascii=False, indent=2))
