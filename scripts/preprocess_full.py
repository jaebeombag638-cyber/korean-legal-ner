import json
import glob
import random
from collections import Counter, defaultdict

from ner_common import (
    get_tokenizer,
    build_section_annotation_map,
    is_drift_annotation,
    section_to_bio,
    LABELS,
)

BASE = "data/12.판결서 익명처리 데이터/3.개방데이터/2.데이터(NIA)"
TRAIN_DIR = f"{BASE}/Training/02.라벨링데이터"
VAL_DIR = f"{BASE}/Validation/02.라벨링데이터"
OUT_DIR = "data/processed"

SEED = 42


def stratified_half_split(files_by_class, seed=SEED):
    """caseClass별로 절반씩 val/test로 나눈다 (분야 비율 유지)."""
    rng = random.Random(seed)
    val_files, test_files = [], []
    for cls, files in files_by_class.items():
        files = list(files)
        rng.shuffle(files)
        half = len(files) // 2
        val_files.extend(files[:half])
        test_files.extend(files[half:])
    return val_files, test_files


def load_caseclass(fp):
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    return data["info"].get("caseClass", "UNKNOWN")


def process_files(files, split_name, tokenizer, stats):
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{split_name}.jsonl"
    n_examples = 0
    label_token_counter = Counter()
    drift_filtered = 0
    total_annotations = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            sec_map = build_section_annotation_map(data)
            case_class = data["info"].get("caseClass", "UNKNOWN")

            for section in data["sections"]:
                sid = section["section_id"]
                text = section["text"]
                anns = sec_map.get(sid, [])
                total_annotations += len(anns)
                for ann in anns:
                    if is_drift_annotation(text, ann):
                        drift_filtered += 1

                input_ids, token_labels = section_to_bio(text, anns, tokenizer=tokenizer)
                for tl in token_labels:
                    if tl != -100:
                        label_token_counter[LABELS[tl]] += 1

                record = {
                    "file": fp,
                    "section_id": sid,
                    "caseClass": case_class,
                    "input_ids": input_ids,
                    "labels": token_labels,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_examples += 1

    stats[split_name] = {
        "n_files": len(files),
        "n_examples": n_examples,
        "label_token_counts": dict(label_token_counter),
        "drift_filtered_annotations": drift_filtered,
        "total_annotations_seen": total_annotations,
    }
    print(f"[{split_name}] files={len(files)} examples={n_examples} -> {out_path}")


def main():
    tokenizer = get_tokenizer()

    train_files = glob.glob(f"{TRAIN_DIR}/**/*.json", recursive=True)
    val_all_files = glob.glob(f"{VAL_DIR}/**/*.json", recursive=True)

    files_by_class = defaultdict(list)
    for fp in val_all_files:
        files_by_class[load_caseclass(fp)].append(fp)

    val_files, test_files = stratified_half_split(files_by_class)

    print(f"train_files={len(train_files)} val_files={len(val_files)} test_files={len(test_files)}")

    stats = {}
    process_files(train_files, "train", tokenizer, stats)
    process_files(val_files, "val", tokenizer, stats)
    process_files(test_files, "test", tokenizer, stats)

    with open(f"{OUT_DIR}/preprocess_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
