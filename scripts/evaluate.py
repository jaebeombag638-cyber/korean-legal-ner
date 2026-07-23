"""4단계: 평가. 3단계에서 선정한 best 모델(기본값 run3)로 test셋을 채점하고
오분류 사례를 뽑아 분석용 파일로 저장한다.
"""
import argparse
import json

import torch
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.data import DataLoader

from ner_common import LABELS, get_tokenizer, id2label


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    input_ids = [r["input_ids"] for r in records]
    labels = [r["labels"] for r in records]
    attention_mask = [[1] * len(ids) for ids in input_ids]
    dataset = Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    )
    return dataset, records


def bio_to_entities(tags):
    """[(start_idx, end_idx_exclusive, type), ...] — 토큰 인덱스 기준."""
    entities = []
    start, cur_type = None, None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                entities.append((start, i, cur_type))
            start, cur_type = i, tag[2:]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            continue
        else:
            if start is not None:
                entities.append((start, i, cur_type))
            start, cur_type = None, None
    if start is not None:
        entities.append((start, len(tags), cur_type))
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="results/run3/best_model")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", default="results/run3")
    parser.add_argument("--max-examples-for-errors", type=int, default=2000,
                         help="오분류 스캔 대상 (전체 9,026건 중 앞부분 N개, 속도용)")
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    dataset, records = load_jsonl(args.test_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    collator = DataCollatorForTokenClassification(tokenizer)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collator)

    all_true, all_pred = [], []
    all_pred_ids = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu()

            for pred_row, label_row in zip(preds, labels):
                true_seq, pred_seq = [], []
                for p, l in zip(pred_row.tolist(), label_row.tolist()):
                    if l == -100:
                        continue
                    true_seq.append(id2label[l])
                    pred_seq.append(id2label[p])
                all_true.append(true_seq)
                all_pred.append(pred_seq)

    report = classification_report(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    print(report)
    print(f"micro F1 = {f1:.4f}")

    with open(f"{args.output_dir}/test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open(f"{args.output_dir}/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_f1": f1}, f, ensure_ascii=False, indent=2)

    # 오분류 사례 수집 (false negative: 못 찾은 개체, false positive: 없는 걸 찾음, wrong type: 위치는 맞았는데 태그 다름)
    mistakes = []
    n_scan = min(args.max_examples_for_errors, len(all_true))
    for idx in range(n_scan):
        true_seq, pred_seq = all_true[idx], all_pred[idx]
        true_ents = set(bio_to_entities(true_seq))
        pred_ents = set(bio_to_entities(pred_seq))

        input_ids = dataset[idx]["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        # -100(special token) 위치를 건너뛰고 true_seq/pred_seq와 같은 길이로 맞춘 토큰 목록
        non_special_tokens = [t for t, l in zip(tokens, dataset[idx]["labels"]) if l != -100]

        record = records[idx]

        for s, e, t in true_ents - pred_ents:
            span_text = tokenizer.convert_tokens_to_string(non_special_tokens[s:e])
            context = tokenizer.convert_tokens_to_string(non_special_tokens[max(0, s - 5):min(len(non_special_tokens), e + 5)])
            mistakes.append({
                "kind": "false_negative (놓침)", "type": t, "span_text": span_text,
                "context": context, "file": record["file"], "section_id": record["section_id"],
            })
        for s, e, t in pred_ents - true_ents:
            span_text = tokenizer.convert_tokens_to_string(non_special_tokens[s:e])
            context = tokenizer.convert_tokens_to_string(non_special_tokens[max(0, s - 5):min(len(non_special_tokens), e + 5)])
            mistakes.append({
                "kind": "false_positive (오탐)", "type": t, "span_text": span_text,
                "context": context, "file": record["file"], "section_id": record["section_id"],
            })

    with open(f"{args.output_dir}/test_misclassified.json", "w", encoding="utf-8") as f:
        json.dump(mistakes[:200], f, ensure_ascii=False, indent=2)

    print(f"오분류 사례 {len(mistakes)}건 수집 (상위 200건 저장: {args.output_dir}/test_misclassified.json)")


if __name__ == "__main__":
    main()
