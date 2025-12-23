# 한국어 심리상담 데이터셋 구축 파이프라인

Gemini 3 API를 활용하여 고품질 한국어 심리상담 데이터셋을 생성하는 파이프라인입니다.

## 주요 특징

- **하이브리드 생산 전략**: Pro 모델(고위험군)과 Flash 모델(일반상담)을 조합한 비용 효율적 생산
- **XAI (설명 가능한 AI)**: Gemini의 Thinking Mode를 활용한 상담사의 임상적 판단 과정 데이터화
- **BLOCK_NONE 설정**: Vertex AI의 안전 필터 제어로 위기 상담 데이터 생성 가능
- **LLM-as-a-Judge**: 자동화된 품질 검증 시스템
- **다양한 카테고리**: 자살 위기부터 진로 상담까지 11개 카테고리 지원

## 프로젝트 구조

```
korean-counseling-dataset/
├── config/
│   ├── settings.py          # 환경설정 (API 키, 프로젝트 ID 등)
│   └── safety_config.py     # 안전 설정 (BLOCK_NONE 등)
├── models/
│   ├── schemas.py           # Pydantic 데이터 스키마
│   └── prompts.py           # 프롬프트 템플릿
├── generators/
│   ├── gemini_client.py     # Gemini API 클라이언트
│   ├── crisis_generator.py  # 고위험군 데이터 생성 (Pro)
│   └── general_generator.py # 일반상담 데이터 생성 (Flash)
├── validators/
│   └── quality_checker.py   # LLM-as-a-Judge 품질 검증
├── pipeline/
│   ├── batch_processor.py   # 배치 처리
│   └── orchestrator.py      # 전체 파이프라인 오케스트레이터
├── data/
│   ├── scenarios/           # 시나리오 템플릿
│   ├── raw/                 # 생성된 원시 데이터
│   ├── validated/           # 검증된 데이터
│   └── final/               # 최종 데이터셋
├── main.py                  # 메인 실행 파일
├── requirements.txt
└── .env.example
```

## 설치

### 1. 저장소 클론

```bash
git clone <repository-url>
cd korean-counseling-dataset
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 Google Cloud 프로젝트 정보 입력
```

### 5. Google Cloud 인증

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## 사용법

### 카테고리 확인

```bash
python main.py --mode list-categories
```

### 단일 세션 생성

```bash
# 자살 위기 상담 세션 생성
python main.py --mode single --track A --category suicide_crisis

# 진로 상담 세션 생성
python main.py --mode single --track B --category career --output ./data/raw/
```

### 배치 생성

```bash
# 일반상담 100건 생성 (품질 검증 포함)
python main.py --mode batch --track B --count 100 --validate

# 고위험군 50건 생성
python main.py --mode batch --track A --count 50 --validate

# 전체 카테고리 생성
python main.py --mode batch --track all --count 200 --validate --output ./data/final/
```

### 전체 파이프라인 실행

```bash
python main.py --mode pipeline --track-a-count 50 --track-b-count 200 --validate
```

### 품질 검증만 실행

```bash
python main.py --mode validate --input data/raw/sessions.jsonl --sample-rate 0.3
```

## 상담 카테고리

### Track A (고위험군 - Gemini Pro)

| 카테고리 | 설명 | 위험 수준 |
|---------|------|----------|
| suicide_crisis | 자살 위기 | Critical |
| self_harm | 자해 충동 | Critical |
| domestic_violence | 가정폭력 | High |
| sexual_assault | 성폭력 피해 | High |
| severe_depression | 심각한 우울증 | High |

### Track B (일반상담 - Gemini Flash)

| 카테고리 | 설명 | 위험 수준 |
|---------|------|----------|
| career | 진로 고민 | Low |
| relationship | 대인관계 | Moderate |
| academic | 학업 스트레스 | Moderate |
| mild_anxiety | 경미한 불안 | Moderate |
| family_conflict | 가족 갈등 | Moderate |
| workplace | 직장 스트레스 | Moderate |

## 데이터 스키마

### CounselingSession (상담 세션)

```json
{
  "session_id": "uuid",
  "category": "suicide_crisis",
  "risk_level": "critical",
  "client_profile": "내담자 프로필",
  "presenting_problem": "주호소 문제",
  "turns": [
    {
      "turn_number": 1,
      "client": {
        "text": "내담자 발화",
        "emotional_state": "감정 상태",
        "risk_indicators": ["위험 지표"],
        "intensity": 8
      },
      "therapist": {
        "clinical_reasoning": "임상적 판단 과정 (XAI)",
        "cognitive_distortions": ["인지 왜곡"],
        "intervention_strategy": "개입 전략",
        "utterance": "상담사 반응",
        "empathy_technique": "반영하기"
      }
    }
  ],
  "session_summary": "세션 요약",
  "clinical_notes": "임상 노트"
}
```

## 품질 검증 기준

| 항목 | 점수/판정 | 설명 |
|-----|---------|------|
| empathy_score | 1-5 | 공감 점수 |
| clinical_appropriateness | pass/fail | 임상적 적절성 |
| korean_naturalness | 1-5 | 한국어 자연스러움 |
| reasoning_coherence | yes/no | 사고과정-답변 일치 |
| safety_check | pass/fail | 안전성 검사 |

## 윤리적 고려사항

### 데이터 보안

- 생성된 데이터는 암호화된 저장소에 보관
- 접근 권한 엄격히 제한
- 고위험군 데이터에 'HIGH_RISK' 태그 필수

### 사용 제한

- 연구 및 교육 목적으로만 사용
- IRB 승인 필수
- 상업적 목적 사용 금지 (별도 라이선스 필요)

### BLOCK_NONE 설정 주의사항

- 오직 통제된 데이터 생성 환경에서만 사용
- 생성된 데이터에는 민감한 내용이 포함될 수 있음
- 후처리를 통한 민감정보 마스킹 권장

## 비용 추정

| 모델 | 용도 | 예상 비용 (1K 세션) |
|-----|------|-------------------|
| Gemini Pro | Track A | ~$20-50 |
| Gemini Flash | Track B | ~$2-5 |

## 라이선스

연구 및 교육 목적 사용 허용. 상업적 사용은 별도 문의.

## 기여

이슈 및 풀 리퀘스트 환영합니다.

## 참고 문헌

- Google Gemini API Documentation
- 한국임상심리학회 상담 가이드라인
- Columbia Suicide Severity Rating Scale
