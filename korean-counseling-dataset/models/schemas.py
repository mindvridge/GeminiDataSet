"""
데이터 스키마 모듈

Pydantic을 사용하여 심리상담 데이터셋의 구조를 정의합니다.
모든 데이터는 이 스키마에 따라 검증되고 직렬화됩니다.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """
    내담자 위험 수준

    상담 데이터의 위험도를 분류하여 적절한 모델과 설정을 선택하는 데 사용됩니다.
    """
    CRITICAL = "critical"      # 자살/자해 위기 - 즉각적 개입 필요
    HIGH = "high"              # 심각한 우울/폭력 - 집중적 관심 필요
    MODERATE = "moderate"      # 중등도 불안/스트레스 - 일반적 상담
    LOW = "low"                # 경미한 고민 - 지지적 상담


class EmpathyTechnique(str, Enum):
    """
    공감 기법

    상담사가 사용하는 핵심 공감 기술을 분류합니다.
    각 기법은 내담자의 감정과 경험을 이해하고 전달하는 데 사용됩니다.
    """
    REFLECTION = "반영하기"           # 내담자의 감정을 거울처럼 되비추기
    RESTATEMENT = "재진술"            # 내담자의 말을 다른 표현으로 바꿔 말하기
    VALIDATION = "타당화"             # 내담자의 감정과 경험이 이해 가능함을 인정
    CLARIFICATION = "명료화"          # 모호한 부분을 명확히 하기
    SUMMARIZATION = "요약"            # 대화 내용을 정리하여 요약
    OPEN_QUESTION = "개방형 질문"      # 깊은 탐색을 유도하는 질문
    NORMALIZATION = "정상화"          # 경험이 정상적임을 알려주기
    AFFIRMATION = "긍정적 강화"        # 강점과 노력을 인정하기
    SELF_DISCLOSURE = "자기개방"       # 적절한 상담사 자기 개방
    SILENCE = "침묵"                  # 의미 있는 침묵 활용


class CounselingCategory(str, Enum):
    """
    상담 카테고리

    Track A (고위험군) - Gemini Pro 사용:
    - suicide_crisis: 자살 위기
    - self_harm: 자해 충동
    - domestic_violence: 가정폭력
    - sexual_assault: 성폭력 피해
    - severe_depression: 심각한 우울증

    Track B (일반상담) - Gemini Flash 사용:
    - career: 진로 고민
    - relationship: 대인관계
    - academic: 학업 스트레스
    - mild_anxiety: 경미한 불안
    - family_conflict: 가족 갈등
    - workplace: 직장 스트레스
    """
    # Track A - 고위험군
    SUICIDE_CRISIS = "suicide_crisis"
    SELF_HARM = "self_harm"
    DOMESTIC_VIOLENCE = "domestic_violence"
    SEXUAL_ASSAULT = "sexual_assault"
    SEVERE_DEPRESSION = "severe_depression"

    # Track B - 일반상담
    CAREER = "career"
    RELATIONSHIP = "relationship"
    ACADEMIC = "academic"
    MILD_ANXIETY = "mild_anxiety"
    FAMILY_CONFLICT = "family_conflict"
    WORKPLACE = "workplace"

    @classmethod
    def get_track(cls, category: "CounselingCategory") -> Literal["A", "B"]:
        """카테고리의 트랙(A/B) 반환"""
        track_a = {
            cls.SUICIDE_CRISIS, cls.SELF_HARM, cls.DOMESTIC_VIOLENCE,
            cls.SEXUAL_ASSAULT, cls.SEVERE_DEPRESSION
        }
        return "A" if category in track_a else "B"

    @classmethod
    def get_korean_name(cls, category: "CounselingCategory") -> str:
        """카테고리의 한글 이름 반환"""
        names = {
            cls.SUICIDE_CRISIS: "자살 위기",
            cls.SELF_HARM: "자해 충동",
            cls.DOMESTIC_VIOLENCE: "가정폭력",
            cls.SEXUAL_ASSAULT: "성폭력 피해",
            cls.SEVERE_DEPRESSION: "심각한 우울증",
            cls.CAREER: "진로 고민",
            cls.RELATIONSHIP: "대인관계",
            cls.ACADEMIC: "학업 스트레스",
            cls.MILD_ANXIETY: "경미한 불안",
            cls.FAMILY_CONFLICT: "가족 갈등",
            cls.WORKPLACE: "직장 스트레스",
        }
        return names.get(category, category.value)


class CognitiveDistortion(str, Enum):
    """
    인지 왜곡 유형

    Aaron Beck의 인지치료 이론에 기반한 대표적인 인지 왜곡 패턴입니다.
    상담사는 이를 식별하고 교정하는 개입을 수행합니다.
    """
    ALL_OR_NOTHING = "흑백논리"              # 전부 아니면 전무
    OVERGENERALIZATION = "과잉일반화"         # 한 가지 사건을 모든 상황에 적용
    MENTAL_FILTER = "정신적 여과"             # 부정적인 것만 걸러서 보기
    DISQUALIFYING_POSITIVE = "긍정 격하"      # 긍정적인 것을 무시하거나 평가절하
    JUMPING_TO_CONCLUSIONS = "성급한 결론"    # 근거 없이 부정적 결론 도출
    MAGNIFICATION = "과대평가"                # 부정적인 것을 확대
    MINIMIZATION = "과소평가"                 # 긍정적인 것을 축소
    EMOTIONAL_REASONING = "감정적 추론"       # 감정을 근거로 판단
    SHOULD_STATEMENTS = "당위적 진술"         # "~해야 한다"는 경직된 규칙
    LABELING = "낙인찍기"                     # 자신이나 타인에게 부정적 꼬리표
    PERSONALIZATION = "개인화"                # 모든 것을 자신의 탓으로 돌림
    CATASTROPHIZING = "파국화"                # 최악의 시나리오만 생각


class ClientUtterance(BaseModel):
    """
    내담자 발화

    상담 세션에서 내담자가 표현한 내용을 구조화합니다.
    감정 상태와 위험 지표를 함께 기록하여 임상적 분석을 지원합니다.
    """
    text: str = Field(
        description="내담자의 발화 내용 (한국어 구어체)"
    )
    emotional_state: str = Field(
        description="내담자의 현재 감정 상태 (예: 우울, 불안, 분노, 무력감)"
    )
    risk_indicators: list[str] = Field(
        default_factory=list,
        description="발견된 위험 지표 (예: 자살 사고, 자해 충동, 희망 상실)"
    )
    intensity: int = Field(
        default=5,
        ge=1,
        le=10,
        description="감정 강도 (1: 매우 약함 ~ 10: 매우 강함)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "요즘 너무 힘들어요. 아무것도 하기 싫고, 그냥 사라지고 싶어요.",
                "emotional_state": "우울, 무력감",
                "risk_indicators": ["희망 상실", "소극적 자살 사고"],
                "intensity": 8
            }
        }


class TherapistResponse(BaseModel):
    """
    상담사 응답 (XAI 포함)

    단순한 응답 텍스트뿐만 아니라, 임상적 판단 과정(사고 흔적)을 포함합니다.
    이는 설명 가능한 AI(XAI) 데이터셋 구축의 핵심 요소입니다.
    """
    clinical_reasoning: str = Field(
        description="임상적 판단 과정 - Gemini의 사고 흔적(Thought Trace)을 기록"
    )
    cognitive_distortions: list[str] = Field(
        default_factory=list,
        description="내담자 발화에서 발견된 인지 왜곡 패턴"
    )
    intervention_strategy: str = Field(
        description="선택한 개입 전략 (예: 위기 개입, 인지 재구조화, 감정 탐색)"
    )
    utterance: str = Field(
        description="내담자에게 전달할 실제 상담 반응 (한국어 구어체, 공감적 화법)"
    )
    empathy_technique: EmpathyTechnique = Field(
        description="주로 사용된 공감 기법"
    )
    secondary_techniques: list[EmpathyTechnique] = Field(
        default_factory=list,
        description="함께 사용된 보조 공감 기법들"
    )
    safety_action: Optional[str] = Field(
        default=None,
        description="안전 관련 조치 (위기 상황 시 필수 기록)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "clinical_reasoning": "내담자의 '사라지고 싶다'는 표현에서 소극적 자살 사고가 의심됨. 즉각적인 위험도 평가가 필요. 먼저 공감적 반영을 통해 내담자가 안전하게 이야기할 수 있는 환경을 조성하고, 이후 구체적인 자살 계획 유무를 탐색해야 함.",
                "cognitive_distortions": ["과잉일반화", "파국화"],
                "intervention_strategy": "위기 개입 - 자살 위험도 평가 및 안전 계획 수립",
                "utterance": "지금 많이 힘드시군요. 사라지고 싶다는 마음이 들 정도로요. 그 마음이 얼마나 무거우실지 조금은 느껴지는 것 같아요. 조금 더 이야기해 주실 수 있을까요?",
                "empathy_technique": "반영하기",
                "secondary_techniques": ["타당화", "개방형 질문"],
                "safety_action": "자살 위험도 평가 시작"
            }
        }


class CounselingTurn(BaseModel):
    """
    상담 대화 턴

    한 턴은 내담자의 발화와 이에 대한 상담사의 응답으로 구성됩니다.
    """
    turn_number: int = Field(
        ge=1,
        description="대화 턴 번호 (1부터 시작)"
    )
    client: ClientUtterance = Field(
        description="내담자 발화"
    )
    therapist: TherapistResponse = Field(
        description="상담사 응답"
    )
    timestamp_offset: int = Field(
        default=0,
        description="세션 시작으로부터의 시간 오프셋 (초)"
    )


class CounselingSession(BaseModel):
    """
    완전한 상담 세션

    하나의 상담 세션을 구성하는 모든 정보를 포함합니다.
    여러 턴의 대화, 임상 노트, 메타데이터 등을 기록합니다.
    """
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="세션 고유 식별자"
    )
    category: CounselingCategory = Field(
        description="상담 카테고리"
    )
    risk_level: RiskLevel = Field(
        description="세션의 전반적인 위험 수준"
    )
    client_profile: str = Field(
        description="내담자 프로필 요약 (익명화된 정보)"
    )
    presenting_problem: str = Field(
        description="주호소 문제 (Main Complaint)"
    )
    turns: list[CounselingTurn] = Field(
        default_factory=list,
        description="상담 대화 턴 목록"
    )
    session_summary: str = Field(
        description="세션 요약"
    )
    clinical_notes: str = Field(
        description="임상 노트 (상담사 소견)"
    )
    treatment_goals: list[str] = Field(
        default_factory=list,
        description="치료 목표"
    )
    homework: Optional[str] = Field(
        default=None,
        description="과제 (다음 세션까지 수행할 활동)"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="추가 메타데이터"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="생성 시각"
    )
    model_used: str = Field(
        default="",
        description="사용된 Gemini 모델"
    )
    generation_config: dict = Field(
        default_factory=dict,
        description="생성 설정"
    )

    @field_validator("turns")
    @classmethod
    def validate_turns(cls, v: list[CounselingTurn]) -> list[CounselingTurn]:
        """턴 번호 순서 검증"""
        for i, turn in enumerate(v, 1):
            if turn.turn_number != i:
                turn.turn_number = i
        return v

    def get_track(self) -> Literal["A", "B"]:
        """세션의 트랙(A/B) 반환"""
        return CounselingCategory.get_track(self.category)

    def to_conversation_text(self) -> str:
        """대화만 추출하여 텍스트로 변환"""
        lines = []
        for turn in self.turns:
            lines.append(f"내담자: {turn.client.text}")
            lines.append(f"상담사: {turn.therapist.utterance}")
        return "\n".join(lines)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "category": "suicide_crisis",
                "risk_level": "critical",
                "client_profile": "20대 후반 여성, 직장인, 최근 실직 경험",
                "presenting_problem": "실직 후 극심한 우울감과 자살 사고 호소",
                "turns": [],
                "session_summary": "내담자는 실직 후 심각한 우울 증상과 자살 사고를 보고함...",
                "clinical_notes": "자살 위험도: 중등도. 구체적 계획은 없으나 지속적 모니터링 필요..."
            }
        }


class QualityScore(BaseModel):
    """
    품질 평가 점수

    LLM-as-a-Judge를 통한 자동화된 품질 검증 결과를 저장합니다.
    """
    session_id: str = Field(
        description="평가 대상 세션 ID"
    )
    empathy_score: int = Field(
        ge=1,
        le=5,
        description="공감 점수 (1: 매우 낮음 ~ 5: 매우 높음)"
    )
    clinical_appropriateness: Literal["pass", "fail"] = Field(
        description="임상적 적절성 (pass: 적절, fail: 부적절)"
    )
    korean_naturalness: int = Field(
        ge=1,
        le=5,
        description="한국어 자연스러움 (1: 번역투 ~ 5: 매우 자연스러움)"
    )
    reasoning_coherence: Literal["yes", "no"] = Field(
        description="사고과정-답변 논리적 일치 여부"
    )
    safety_check: Literal["pass", "fail"] = Field(
        description="안전성 검사 (위기 상황 적절 대응 여부)"
    )
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="종합 점수 (0.0 ~ 1.0)"
    )
    feedback: str = Field(
        description="평가자 피드백"
    )
    evaluated_at: datetime = Field(
        default_factory=datetime.now,
        description="평가 시각"
    )
    evaluator_model: str = Field(
        default="gemini-pro",
        description="평가에 사용된 모델"
    )

    def is_acceptable(self, min_empathy: int = 3, min_naturalness: int = 3) -> bool:
        """품질 기준 충족 여부 확인"""
        return (
            self.empathy_score >= min_empathy
            and self.korean_naturalness >= min_naturalness
            and self.clinical_appropriateness == "pass"
            and self.safety_check == "pass"
        )


class BatchResult(BaseModel):
    """
    배치 처리 결과

    배치 데이터 생성의 결과를 요약합니다.
    """
    batch_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="배치 고유 식별자"
    )
    total_requested: int = Field(
        description="요청된 총 데이터 수"
    )
    total_generated: int = Field(
        default=0,
        description="생성 성공한 데이터 수"
    )
    total_failed: int = Field(
        default=0,
        description="생성 실패한 데이터 수"
    )
    total_validated: int = Field(
        default=0,
        description="검증 통과한 데이터 수"
    )
    total_rejected: int = Field(
        default=0,
        description="검증 탈락한 데이터 수"
    )
    categories: dict[str, int] = Field(
        default_factory=dict,
        description="카테고리별 생성 현황"
    )
    average_quality_score: float = Field(
        default=0.0,
        description="평균 품질 점수"
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        description="예상 비용 (USD)"
    )
    duration_seconds: float = Field(
        default=0.0,
        description="소요 시간 (초)"
    )
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="시작 시각"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="완료 시각"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="발생한 오류 목록"
    )

    def success_rate(self) -> float:
        """성공률 계산"""
        if self.total_requested == 0:
            return 0.0
        return self.total_generated / self.total_requested

    def validation_rate(self) -> float:
        """검증 통과율 계산"""
        if self.total_generated == 0:
            return 0.0
        return self.total_validated / self.total_generated


class GenerationRequest(BaseModel):
    """
    데이터 생성 요청

    단일 또는 배치 데이터 생성을 위한 요청 정보를 담습니다.
    """
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="요청 고유 식별자"
    )
    category: CounselingCategory = Field(
        description="생성할 상담 카테고리"
    )
    count: int = Field(
        default=1,
        ge=1,
        description="생성할 데이터 수"
    )
    min_turns: int = Field(
        default=5,
        ge=3,
        description="최소 대화 턴 수"
    )
    max_turns: int = Field(
        default=10,
        le=20,
        description="최대 대화 턴 수"
    )
    include_crisis: bool = Field(
        default=False,
        description="위기 상황 포함 여부 (Track A 전용)"
    )
    few_shot_examples: list[CounselingSession] = Field(
        default_factory=list,
        description="Few-shot 예시 (Flash 모델용)"
    )
    custom_scenario: Optional[str] = Field(
        default=None,
        description="커스텀 시나리오 (선택적)"
    )
    validate_output: bool = Field(
        default=True,
        description="생성 후 품질 검증 실행 여부"
    )

    def get_track(self) -> Literal["A", "B"]:
        """요청의 트랙(A/B) 반환"""
        return CounselingCategory.get_track(self.category)
