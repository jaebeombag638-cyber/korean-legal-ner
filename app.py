"""5단계: Streamlit 데모 앱. Run3 best_model로 텍스트 속 개인정보(PER/LOC/ORG)를 탐지·익명화한다.
계획.md 5단계 스펙: (1) entity 하이라이트 (2) 익명화된 텍스트 (3) 탐지 entity 표.
"""
import os

# Streamlit은 스크립트를 별도 스레드(ScriptRunner)에서 실행하는데, 이 스레드에서
# torch/tokenizers의 네이티브 스레드풀(OpenMP, Rust rayon)을 초기화하면 Windows에서
# 세그폴트(exit 139)가 발생하는 걸 확인했다 — 반드시 torch/transformers import 이전에
# 단일 스레드로 강제해야 한다.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import html

import torch

torch.set_num_threads(1)

import streamlit as st
from transformers import pipeline

MODEL_DIR = "results/run3/best_model"

TAG_COLOR = {"PER": "#ffd166", "LOC": "#06d6a0", "ORG": "#a0c4ff"}
TAG_LABEL_KO = {"PER": "인명", "LOC": "주소", "ORG": "기관명"}

EXAMPLE_TEXT = (
    "피고 최도선은 부산시 중구에서 대전지방법원 판결을 받았다. "
    "원고 측 대리인은 서울지방법원 남부지원 소속이다."
)


@st.cache_resource
def load_ner():
    return pipeline(
        "ner",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        aggregation_strategy="simple",
    )


def run_ner(ner, text):
    """entity_group이 LABEL_1처럼 나오는 경우를 대비해 id2label 그대로 신뢰 (config.json에 PER/LOC/ORG로 저장됨).
    태그별 순번(label, 예: PER-1)을 여기서 한 번만 매겨서, 하이라이트·익명화 텍스트·표가
    항상 같은 번호를 가리키게 한다."""
    results = ner(text)
    entities = []
    for r in results:
        tag = r["entity_group"]
        if tag not in TAG_COLOR:
            continue
        entities.append({
            "start": r["start"],
            "end": r["end"],
            "tag": tag,
            "text": text[r["start"]:r["end"]],
            "score": float(r["score"]),
        })
    entities.sort(key=lambda e: e["start"])

    counters = {"PER": 0, "LOC": 0, "ORG": 0}
    for e in entities:
        counters[e["tag"]] += 1
        e["label"] = f'{e["tag"]}-{counters[e["tag"]]}'

    return entities


def render_highlighted(text, entities):
    """탐지된 entity를 태그별 색상으로 하이라이트한 HTML을 만든다."""
    pieces = []
    cursor = 0
    for e in entities:
        if e["start"] < cursor:
            continue  # 겹치는 span은 건너뜀 (aggregation_strategy=simple이라 거의 없음)
        pieces.append(html.escape(text[cursor:e["start"]]))
        color = TAG_COLOR[e["tag"]]
        span_text = html.escape(e["text"])
        pieces.append(
            f'<span style="background-color:{color};border-radius:4px;padding:0 2px;">'
            f'{span_text}<sub style="font-size:0.7em;">{e["tag"]}</sub></span>'
        )
        cursor = e["end"]
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces).replace("\n", "<br>")


def render_entity_table_md(entities):
    """st.dataframe/st.table은 이 환경에서 pyarrow와 torch가 같이 로드되면
    네이티브 세그폴트(exit 139)를 일으키는 걸 확인해, pyarrow를 쓰지 않는
    마크다운 표로 대체한다."""
    rows = ["| 태그 | 원문 | 대체 텍스트 | 신뢰도 |", "|---|---|---|---|"]
    for e in sorted(entities, key=lambda x: x["start"]):
        rows.append(f'| {e["tag"]} | {e["text"]} | {e["label"]} | {e["score"]:.3f} |')
    return "\n".join(rows)


def render_anonymized(text, entities):
    """탐지된 entity를 run_ner()에서 매긴 순번 placeholder([PER-1] 등)로 치환한다."""
    pieces = []
    cursor = 0
    for e in entities:
        if e["start"] < cursor:
            continue
        pieces.append(text[cursor:e["start"]])
        pieces.append(f'[{e["label"]}]')
        cursor = e["end"]
    pieces.append(text[cursor:])
    return "".join(pieces)


st.set_page_config(page_title="판결문 개인정보 익명화", page_icon="🔒")
st.title("한국어 판결문 개인정보 자동 익명화")
st.caption(
    "KLUE-BERT NER 모델(Run3, test F1 0.9045)로 인명(PER)·주소(LOC)·기관명(ORG)을 탐지합니다. "
    "판결문 특화 학습 데이터를 사용했으므로 다른 도메인 텍스트에서는 성능이 다를 수 있습니다."
)

text = st.text_area("텍스트 입력", value=EXAMPLE_TEXT, height=200)

if st.button("익명화 실행", type="primary"):
    if not text.strip():
        st.warning("텍스트를 입력해주세요.")
    else:
        with st.spinner("모델 로딩 및 추론 중..."):
            ner = load_ner()
            entities = run_ner(ner, text)

        st.subheader("탐지 결과 하이라이트")
        st.markdown(render_highlighted(text, entities), unsafe_allow_html=True)
        legend = "  ".join(
            f'<span style="background-color:{c};border-radius:4px;padding:0 4px;">{TAG_LABEL_KO[t]}({t})</span>'
            for t, c in TAG_COLOR.items()
        )
        st.markdown(legend, unsafe_allow_html=True)

        st.subheader("익명화된 텍스트")
        st.text(render_anonymized(text, entities))

        st.subheader(f"탐지된 개체 목록 ({len(entities)}건)")
        if entities:
            st.markdown(render_entity_table_md(entities))
        else:
            st.info("탐지된 개인정보가 없습니다.")

st.divider()
st.caption(
    "⚠️ 포트폴리오 목적의 데모입니다. PER F1 0.55로 인명 탐지 성능이 상대적으로 약하고, "
    "중첩된 기관명의 경계 오류가 있을 수 있습니다 (자세한 한계는 README.md 참조)."
)
