"""3단계: 모델 학습. klue/bert-base를 NER용으로 파인튜닝한다.
계획.md 3단계 실험 표(Run1~Run4)를 --lr/--epochs/--class-weight 인자로 재현한다.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ner_common import LABELS, get_tokenizer, id2label, label2id

DATA_DIR = "data/processed"


def load_jsonl(path):
    input_ids, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            input_ids.append(d["input_ids"])
            labels.append(d["labels"])
    attention_mask = [[1] * len(ids) for ids in input_ids]
    return Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    )


def compute_class_weights(train_dataset):
    """태그 빈도 역비례 가중치 (2단계에서 발견한 ORG:PER:LOC 불균형 대응, 계획.md 3단계 참조)."""
    counts = np.zeros(len(LABELS))
    for labels in train_dataset["labels"]:
        for l in labels:
            if l != -100:
                counts[l] += 1
    counts[counts == 0] = 1
    weights = counts.sum() / (len(LABELS) * counts)
    return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_compute_metrics():
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels, pred_labels = [], []
        for pred_row, label_row in zip(predictions, labels):
            true_seq, pred_seq = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                true_seq.append(id2label[l])
                pred_seq.append(id2label[p])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq)

        return {
            "f1": f1_score(true_labels, pred_labels),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="run1")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--train-file", default=f"{DATA_DIR}/train.jsonl")
    parser.add_argument("--val-file", default=f"{DATA_DIR}/val.jsonl")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--max-train-examples", type=int, default=None,
                         help="스모크 테스트용 — 지정 시 train 앞부분만 사용")
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    train_dataset = load_jsonl(args.train_file)
    val_dataset = load_jsonl(args.val_file)
    if args.max_train_examples:
        train_dataset = train_dataset.select(range(min(args.max_train_examples, len(train_dataset))))

    class_weights = compute_class_weights(train_dataset) if args.class_weight else None

    model = AutoModelForTokenClassification.from_pretrained(
        "klue/bert-base",
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label,
    )

    output_dir = f"{args.output_root}/{args.run_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
        class_weights=class_weights,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    best_model_dir = f"{output_dir}/best_model"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    predictions, labels, _ = trainer.predict(val_dataset)
    predictions = np.argmax(predictions, axis=2)
    true_labels, pred_labels = [], []
    for pred_row, label_row in zip(predictions, labels):
        true_seq, pred_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            true_seq.append(id2label[l])
            pred_seq.append(id2label[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

    report = classification_report(true_labels, pred_labels)
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
    with open(f"{output_dir}/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(json.dumps(eval_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
