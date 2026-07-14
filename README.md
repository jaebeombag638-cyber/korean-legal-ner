# 판결문 기반 한국어 개인정보 자동 익명화

![상태](https://img.shields.io/badge/상태-진행중-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E)

AI Hub 판결문 익명화 데이터로 **KLUE-BERT**를 파인튜닝하여, 한국어 법률 문서에서 개인정보(이름·주소·날짜·기관명 등)를 자동으로 탐지·교체하는 시스템을 구축합니다.

---

## 핵심 질문

> **경량 모델(110M 파라미터)로 공식 벤치마크(F1 0.9589)에 얼마나 근접할 수 있는가?**

공식 벤치마크는 GPT-OSS-120B(QLoRA)로 달성한 수치입니다.  
이 프로젝트는 KLUE-BERT 수준의 경량 모델로 **F1 0.80 이상**을 목표로 하며, 그 한계를 솔직하게 분석하는 것이 핵심입니다.

---

## 동작 예시

```
입력:  "피고 최도선은 2021년 3월 부산시 중구에서 사건을 일으켰다."

탐지:
  최도선    → [PER]  → 증인-1
  2021년 3월 → [DAT]  → 2021년 XX월
  부산시 중구 → [LOC]  → 부산광역시 중구 A

출력:  "피고 증인-1은 2021년 XX월 부산광역시 중구 A에서 사건을 일으켰다."
```

---

## 기술 스택

| 구분 | 도구 |
|------|------|
| 언어 모델 | [KLUE-BERT](https://huggingface.co/klue/bert-base) (klue/bert-base) |
| 학습 프레임워크 | HuggingFace Transformers + PyTorch |
| 평가 | seqeval (entity-level F1) |
| 데모 앱 | Streamlit (Hugging Face Spaces 배포 예정) |
| 학습 환경 | 로컬 GPU (RTX 3060, 12GB) — Colab T4 대체 |

---

## 데이터셋

**출처**: [AI Hub — 판결서 익명처리 데이터셋](https://aihub.or.kr)

| 구분 | 건수 |
|------|------|
| Training (train) | 일반판결 12,000 + 1·2·최종심 4,800 |
| Validation → val | 일반판결 750 + 1·2·최종심 300 |
| Validation → test | 일반판결 750 + 1·2·최종심 300 |
| AI Hub 비공개 test | 2,100 (리더보드용, 미사용) |

10개 법률 분야(민사·형사A/B·가사·일반행정·근로자·특허·금융조세·기업·개인정보)를 포함합니다.

---

## NER 태그 체계

익명화 규칙을 NER 태그로 매핑하여 BIO 방식으로 학습합니다.

| 태그 | 대상 | 익명화 예시 |
|------|------|------------|
| **PER** | 인명 (R1) | `최도선` → `증인-1` |
| **LOC** | 주소 (R3) | `부산시 중구 광복동` → `부산광역시 중구 A` |
| **DAT** | 날짜 (R4) | `1961.12.20.` → `1961. XX. XX.` |
| **ORG** | 기관명 (R7) | `대전지방법원` → `조직-2` |
| **ID** | 주민번호·사업자번호 (R2, R8) | `800101-1234567` → 삭제 |

> R5, R6, R9, R10은 EDA 후 실제 등장 여부를 확인하여 태그를 추가합니다.

---

## 프로젝트 구조

```
korean-legal-ner/
├── data/                  # AI Hub 원본 데이터 (git 제외)
├── notebooks/
│   ├── 01_eda.ipynb       # 규칙 분포 분석 및 태그 매핑 확정
│   ├── 02_preprocess.ipynb # BIO 변환 및 전처리 검증
│   ├── 03_train.ipynb     # 모델 학습 (Colab)
│   └── 04_eval.ipynb      # 평가 및 오분류 분석
├── app.py                 # Streamlit 데모 앱
├── 계획.md                # 상세 구현 계획
└── 개념.md                # 프로젝트 핵심 개념 정리
```

---

## 구현 로드맵

- [x] **0단계** — 환경 세팅 (로컬 GPU 사용, Colab 대체)
- [ ] **1단계** — EDA: 규칙 분포 확인 → NER 태그 매핑 확정
- [ ] **2단계** — 전처리: span → BIO 변환, `verify_span` 100% 통과 검증
- [ ] **3단계** — 모델 학습: learning rate·epoch 3회 실험
- [ ] **4단계** — 평가: entity별 F1, 오분류 20건 이상 분석
- [ ] **5단계** — Streamlit 데모 앱 구현 및 Hugging Face Spaces 배포
- [ ] **6단계** — 결론: 벤치마크 비교 및 한계점 정리

---

## 목표 성능

| 모델 | 파라미터 | F1 | 비고 |
|------|---------|-----|------|
| GPT-OSS-120B (QLoRA) | 120B | **0.9589** | AI Hub 공식 벤치마크, 비공개 test셋 기준 |
| **KLUE-BERT (본 프로젝트)** | 110M | **0.80 목표** | Validation에서 직접 분리한 test셋 기준 |

> 두 모델의 test셋이 다르므로 수치 비교는 참고용입니다. 조건 차이는 결론에서 명시합니다.

---

## 설치 및 실행

> 진행 중 — 단계별 완료 시 업데이트됩니다.

```bash
pip install transformers datasets seqeval torch streamlit
```

```bash
# 데모 앱 실행 (5단계 완료 후)
streamlit run app.py
```

---

## 알려진 한계

- **512 토큰 초과 섹션** (실측 약 21%): truncation으로 후반부 annotation 학습 누락
- **R4 날짜 분기**: 다양한 날짜 표기 패턴을 단순화 처리 시 오분류 발생
- **사법인물 예외**: 판사·검사 등 공직자 이름의 익명화 예외 처리 완성도 한계
- **동일 test셋 부재**: 공식 벤치마크(비공개 test셋)와 직접 비교 불가
- **span 위치 불일치** (샘플 검사 약 10.4%): `section_text[start:end]`가 `original_text`와 어긋나는 annotation이 존재. 원인은 (1) 원문이 이미 `#이름#` 형태 placeholder로 치환된 경우, (2) annotation 계산 시점과 배포된 텍스트 버전이 달라 위치가 밀린 경우 두 가지로 추정. 학습 시 불일치 annotation은 필터링하여 제외

---

## 라이선스

본 프로젝트는 포트폴리오 목적으로 제작되었습니다.  
데이터는 AI Hub 이용약관에 따르며 재배포하지 않습니다.