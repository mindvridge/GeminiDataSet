"""
한국어 심리상담 데이터셋 - 데이터 모델 모듈

이 모듈은 데이터 스키마와 프롬프트 템플릿을 정의합니다:
- schemas: Pydantic 기반 데이터 스키마
- prompts: 상담 시나리오 생성을 위한 프롬프트 템플릿
"""

from .schemas import (
    RiskLevel,
    EmpathyTechnique,
    CounselingCategory,
    ClientUtterance,
    TherapistResponse,
    CounselingTurn,
    CounselingSession,
    QualityScore,
    BatchResult,
    GenerationRequest,
)

from .prompts import (
    PromptTemplate,
    get_system_prompt,
    get_scenario_prompt,
    get_few_shot_examples,
    TRACK_A_CATEGORIES,
    TRACK_B_CATEGORIES,
)

__all__ = [
    # Schemas
    "RiskLevel",
    "EmpathyTechnique",
    "CounselingCategory",
    "ClientUtterance",
    "TherapistResponse",
    "CounselingTurn",
    "CounselingSession",
    "QualityScore",
    "BatchResult",
    "GenerationRequest",
    # Prompts
    "PromptTemplate",
    "get_system_prompt",
    "get_scenario_prompt",
    "get_few_shot_examples",
    "TRACK_A_CATEGORIES",
    "TRACK_B_CATEGORIES",
]
