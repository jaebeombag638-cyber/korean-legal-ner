# 판결문 기반 한국어 개인정보 자동 익명화

![상태](https://img.shields.io/badge/상태-진행중-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E)

AI Hub 판결문 익명화 데이터로 **KLUE-BERT**를 파인튜닝하여, 한국어 법률 문서에서 개인정보(이름·주소·기관명)를 자동으로 탐지·교체하는 시스템을 구축합니다.

> 1단계 EDA 결과, 공개 데이터셋에는 인명(R1)·주소(R3)·기관명(R7) annotation만 존재하고 날짜(R4)·주민번호(R2/R8) 등은 없는 것으로 확인되어, 태그 범위를 PER·LOC·ORG 세 가지로 좁혔습니다 (자세한 내용은 [진행일지.md](진행일지.md) 1단계 참조).

---

## 핵심 질문

> **경량 모델(110M 파라미터)로 공식 벤치마크(F1 0.9589)에 얼마나 근접할 수 있는가?**

공식 벤치마크는 GPT-OSS-120B(QLoRA)로 달성한 수치입니다.  
이 프로젝트는 KLUE-BERT 수준의 경량 모델로 **F1 0.80 이상**을 목표로 하며, 그 한계를 솔직하게 분석하는 것이 핵심입니다.

---

## 동작 예시

```
입력:  "피고 최도선은 부산시 중구에서 대전지방법원 판결을 받았다."

탐지:
  최도선    → [PER]  → 증인-1
  부산시 중구 → [LOC]  → 부산광역시 중구 A
  대전지방법원 → [ORG]  → 조직-1

출력:  "피고 증인-1은 부산광역시 중구 A에서 조직-1 판결을 받았다."
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
| **ORG** | 기관명 (R7) | `대전지방법원` → `조직-2` |

> **1단계 EDA 결과**: 전체 18,900개 파일(annotation 341,355건)을 전수 조사한 결과 R1·R3·R7만 실제로 존재했고, 공식 문서가 언급한 R2(주민번호)·R4(날짜)·R8(사업자번호)와 미공개였던 R5·R6·R9·R10은 **단 한 건도 없었습니다**. 이에 따라 DAT·ID 태그는 목표에서 제외했습니다 (원래 5개 태그 계획이었으나 실제 학습 가능한 것은 3개). 자세한 수치는 [진행일지.md](진행일지.md)와 [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) 참조.

---

## 프로젝트 구조

```
korean-legal-ner/
├── data/                  # AI Hub 원본 데이터 + 전처리 결과 (git 제외)
│   └── processed/         # 2단계 산출물: train/val/test.jsonl
├── scripts/
│   ├── ner_common.py      # BIO 변환 핵심 함수 (노트북과 공유)
│   ├── eda_full_scan.py   # 1단계 전체 데이터 분석
│   └── preprocess_full.py # 2단계 전체 데이터 전처리
├── notebooks/
│   ├── 01_eda.ipynb       # 규칙 분포 분석 및 태그 매핑 확정
│   ├── 02_preprocess.ipynb # BIO 변환 및 전처리 검증
│   ├── 03_train.ipynb     # 모델 학습
│   └── 04_eval.ipynb      # 평가 및 오분류 분석
├── app.py                 # Streamlit 데모 앱
├── 계획.md                # 상세 구현 계획
├── 개념.md                # 프로젝트 핵심 개념 정리
├── official.md            # AI Hub 공식 문서 정리 + 실측 검증 결과
└── 진행일지.md            # 단계별 진행 기록 (초보자용)
```

---

## 구현 로드맵

- [x] **0단계** — 환경 세팅 (로컬 GPU 사용, Colab 대체)
- [x] **1단계** — EDA: 규칙 분포 확인 → NER 태그 매핑 확정 (PER·LOC·ORG 3종)
- [x] **2단계** — 전처리: span → BIO 변환, drift 불일치(7.86%) 필터링, train/val/test 생성
- [x] **3단계** — 모델 학습: learning rate·epoch·class weight 4회 실험 완료. Run1~3(F1 0.9071~0.9077)은 사실상 동급, Run4(class weight)는 오히려 F1 0.8230으로 하락 — **Run3(lr=2e-5, epoch=3)을 최종 baseline으로 선정**, PER F1(0.47~0.49)이 ORG(0.93) 대비 약한 점은 미해결 (자세한 내용은 [진행일지.md](진행일지.md) 3단계 참조)
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

- **512 토큰 초과 섹션** (전체 18,900개 파일 실측 13.29%): truncation으로 후반부 annotation 학습 누락
- **태그 범위 축소**: 공개 데이터에 R2(주민번호)·R4(날짜)·R8(사업자번호)·R5·R6·R9·R10이 전혀 없어 DAT·ID 탐지는 애초에 학습 불가 — PER·LOC·ORG 3종만 지원
- **사법인물 예외**: 판사·검사 등 공직자 이름의 익명화 예외 처리 완성도 한계
- **동일 test셋 부재**: 공식 벤치마크(비공개 test셋)와 직접 비교 불가
- **span 위치 불일치** (전체 341,355건 중 13.09%): `section_text[start:end]`가 `original_text`와 어긋나는 annotation이 존재. 두 가지로 나뉨 — (1) 원문이 이미 `#이름#` 형태 placeholder로 치환된 경우(5.23%): 위치 자체는 정확해 그대로 학습에 사용, (2) annotation 계산 시점과 배포된 텍스트 버전이 달라 위치가 완전히 밀린 경우(7.86%): 잘못된 라벨 주입을 막기 위해 제외. 실제 제외되는 비율은 7.86%뿐

---

## 라이선스

본 프로젝트는 포트폴리오 목적으로 제작되었습니다.  
데이터는 AI Hub 이용약관에 따르며 재배포하지 않습니다.