# Korean Smishing Detector

한국어 문자 메시지(SMS) 데이터를 활용한 스미싱 탐지 머신러닝 프로젝트입니다.

## 프로젝트 개요

스미싱(Smishing)은 SMS + Phishing의 합성어로, 문자 메시지를 통한 피싱 공격입니다.  
본 프로젝트는 머신러닝 기법을 활용해 스미싱 문자와 정상 문자를 자동으로 분류합니다.

## 사용 데이터

- **출처**: AI Hub — 스팸·스미싱 문자 데이터셋
- **언어**: 한국어
- **분류**: 스미싱 / 정상 (이진 분류)

## 기술 스택

- Python 3.x
- pandas, numpy
- KoNLPy (Okt) — 한국어 형태소 분석
- scikit-learn — TF-IDF, 모델 학습·평가
- matplotlib, seaborn — 시각화
- Jupyter Notebook

## 진행 계획

| 단계 | 내용 |
|------|------|
| 1. 데이터 탐색 (EDA) | 데이터 분포 확인, 스미싱 키워드 시각화 |
| 2. 전처리 | 형태소 분석 + TF-IDF 벡터화 |
| 3. 모델 학습·비교 | Logistic Regression, Naive Bayes, SVM |
| 4. 성능 평가 | F1 Score, Confusion Matrix, 오분류 사례 분석 |
| 5. 결론 | 모델별 비교 및 인사이트 정리 |

## 프로젝트 구조

korean-smishing-detector/
├── data/           # 원본 데이터 (AI Hub 다운로드)
├── notebooks/      # Jupyter 분석 노트북
└── README.md

## 진행 기간

약 1주일 (개인 학습 목적)