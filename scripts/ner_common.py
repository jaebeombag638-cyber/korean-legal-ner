"""2단계 이후 여러 스크립트/노트북에서 공유하는 핵심 로직.
계획.md의 코드와 동일한 내용을 실제로 동작하는 모듈로 옮겨 둔 것.
"""

from transformers import AutoTokenizer

MODEL_NAME = "klue/bert-base"

# 1단계 EDA로 확정: 공개 데이터에는 R1·R3·R7만 존재 (R2/R4/R5/R6/R8/R9/R10은 0건)
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

RULE_TO_TAG = {
    "R1": "PER",
    "R3": "LOC",
    "R7": "ORG",
}

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def build_section_annotation_map(data):
    """JSON 1개에서 section_id → [annotation 리스트] 매핑 생성."""
    mapping = {s["section_id"]: [] for s in data["sections"]}
    for ann in data.get("annotations", []):
        for sid in ann["section_id"]:
            if sid in mapping:
                mapping[sid].append(ann)
    return mapping


def is_drift_annotation(section_text, ann):
    """span 불일치 중 'drift'(위치 자체가 다른 텍스트)만 True.
    'placeholder' 불일치(#이름# 등)는 위치는 맞으므로 False (학습에 사용)."""
    for span in ann["span"]:
        extracted = section_text[span["start"]:span["end"]]
        orig = ann["original_text"]
        if extracted == orig:
            continue
        if "#" in orig or "#" in extracted:
            continue
        return True
    return False


def section_to_bio(section_text, annotations_in_section, tokenizer=None):
    """섹션 텍스트 + annotations → (input_ids, token_labels)."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    char_labels = ["O"] * len(section_text)

    for ann in annotations_in_section:
        tag = RULE_TO_TAG.get(ann["rules_triggered"], "O")
        if tag == "O":
            continue
        if is_drift_annotation(section_text, ann):
            continue
        for span in ann["span"]:
            start, end = span["start"], span["end"]
            if start >= len(section_text) or end > len(section_text) or start >= end:
                continue
            char_labels[start] = f"B-{tag}"
            for i in range(start + 1, end):
                char_labels[i] = f"I-{tag}"

    encoding = tokenizer(
        section_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )
    token_labels = []
    for token_start, token_end in encoding["offset_mapping"]:
        if token_start == 0 and token_end == 0:
            token_labels.append(-100)
        else:
            token_labels.append(label2id[char_labels[token_start]])

    return encoding["input_ids"], token_labels


def verify_span(section_text, ann):
    """진단용: original_text가 span 위치에 정확히 있는지 (완전 일치 기준)."""
    for span in ann["span"]:
        extracted = section_text[span["start"]:span["end"]]
        if extracted != ann["original_text"]:
            return False
    return True
