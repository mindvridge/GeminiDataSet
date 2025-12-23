"""
안전 설정 모듈 (Safety Configuration)

Vertex AI의 안전 필터 설정을 관리합니다.
임상 데이터 생성을 위해 BLOCK_NONE 설정을 구현합니다.

주의사항:
=========
이 설정은 오직 통제된 연구 환경에서의 데이터 생성 목적으로만 사용되어야 합니다.
생성된 데이터는 암호화된 저장소에 보관하고, 접근 권한을 엄격히 제한해야 합니다.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

# google-genai SDK의 types 모듈
# 실제 사용 시에는 google.genai.types를 import 해야 합니다.
try:
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    types = None


class HarmCategory(str, Enum):
    """
    Gemini API 유해 카테고리

    Vertex AI에서 지원하는 4가지 유해 카테고리입니다.
    각 카테고리에 대해 필터링 임계값을 설정할 수 있습니다.
    """
    HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
    SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    HARASSMENT = "HARM_CATEGORY_HARASSMENT"


class HarmBlockThreshold(str, Enum):
    """
    유해 콘텐츠 차단 임계값

    BLOCK_NONE: 필터링 없음 (임상 데이터 생성용)
    BLOCK_ONLY_HIGH: 높은 확률의 유해 콘텐츠만 차단
    BLOCK_MEDIUM_AND_ABOVE: 중간 이상의 유해 콘텐츠 차단
    BLOCK_LOW_AND_ABOVE: 낮은 이상의 유해 콘텐츠 차단 (가장 엄격)
    """
    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"


@dataclass
class SafetySetting:
    """단일 안전 설정"""
    category: HarmCategory
    threshold: HarmBlockThreshold


@dataclass
class SafetyConfig:
    """
    전체 안전 설정 구성

    트랙(A/B)에 따라 다른 안전 설정을 적용합니다.
    - Track A (고위험군): 모든 필터 해제 (BLOCK_NONE)
    - Track B (일반상담): 높은 위험만 차단 (BLOCK_ONLY_HIGH)
    """
    settings: list[SafetySetting]
    track: Literal["A", "B"]
    description: str

    def to_genai_settings(self) -> list:
        """
        google-genai SDK 형식의 SafetySetting 리스트 반환

        Returns:
            types.SafetySetting 객체 리스트
        """
        if not GENAI_AVAILABLE:
            # SDK가 설치되지 않은 경우 딕셔너리 형태로 반환
            return [
                {
                    "category": setting.category.value,
                    "threshold": setting.threshold.value
                }
                for setting in self.settings
            ]

        # google-genai SDK의 types.SafetySetting 사용
        return [
            types.SafetySetting(
                category=setting.category.value,
                threshold=setting.threshold.value
            )
            for setting in self.settings
        ]

    def to_dict_list(self) -> list[dict]:
        """
        딕셔너리 리스트 형태로 반환 (디버깅/로깅용)

        Returns:
            안전 설정 딕셔너리 리스트
        """
        return [
            {
                "category": setting.category.value,
                "threshold": setting.threshold.value
            }
            for setting in self.settings
        ]


def get_crisis_safety_settings() -> SafetyConfig:
    """
    위기 상담 데이터 생성을 위한 안전 설정 (Track A)

    모든 유해 카테고리에 대해 필터링을 비활성화(BLOCK_NONE)합니다.
    이 설정은 자살 위기, 자해 충동, 가정폭력 등 고위험 시나리오의
    데이터를 생성할 때 사용됩니다.

    주의:
    -----
    - 이 설정은 오직 통제된 데이터 생성 환경에서만 사용되어야 합니다.
    - 생성된 데이터는 암호화된 저장소에 보관해야 합니다.
    - 접근 권한을 엄격히 제한해야 합니다.
    - Vertex AI의 BLOCK_NONE 사용 권한이 필요합니다.

    Returns:
        SafetyConfig: 모든 필터가 해제된 안전 설정
    """
    settings = [
        SafetySetting(
            category=HarmCategory.HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=HarmCategory.DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=HarmCategory.SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=HarmCategory.HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE
        ),
    ]

    return SafetyConfig(
        settings=settings,
        track="A",
        description="위기 상담 데이터 생성용 - 모든 안전 필터 해제"
    )


def get_general_safety_settings() -> SafetyConfig:
    """
    일반 상담 데이터 생성을 위한 안전 설정 (Track B)

    높은 확률의 유해 콘텐츠만 차단(BLOCK_ONLY_HIGH)합니다.
    진로, 대인관계, 학업 스트레스 등 일반적인 상담 시나리오에 사용됩니다.

    Returns:
        SafetyConfig: 높은 위험만 차단하는 안전 설정
    """
    settings = [
        SafetySetting(
            category=HarmCategory.HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        SafetySetting(
            category=HarmCategory.DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        SafetySetting(
            category=HarmCategory.SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        SafetySetting(
            category=HarmCategory.HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
    ]

    return SafetyConfig(
        settings=settings,
        track="B",
        description="일반 상담 데이터 생성용 - 높은 위험만 차단"
    )


def get_safety_settings(track: Literal["A", "B"] = "B") -> SafetyConfig:
    """
    트랙에 따른 안전 설정 반환

    Args:
        track: "A" (고위험군) 또는 "B" (일반상담)

    Returns:
        SafetyConfig: 해당 트랙의 안전 설정
    """
    if track == "A":
        return get_crisis_safety_settings()
    return get_general_safety_settings()


# ==================== 데이터 보안 가이드라인 ====================

DATA_SECURITY_GUIDELINES = """
===========================================
    데이터 보안 및 윤리 가이드라인
===========================================

1. 데이터 저장 및 암호화
   - 생성된 모든 원시 데이터는 암호화된 GCS 버킷에 저장
   - 로컬 저장 시 AES-256 암호화 적용
   - 전송 시 TLS 1.3 이상 사용

2. 접근 권한 관리
   - 인가된 연구원만 데이터에 접근 가능
   - 역할 기반 접근 제어(RBAC) 적용
   - 모든 접근 로그 기록

3. 데이터 레이블링
   - 고위험군 데이터에는 'HIGH_RISK' 태그 필수
   - 생성 일시, 모델 버전, 생성자 정보 기록
   - 감사 추적(Audit Trail) 유지

4. 데이터 보존 정책
   - 연구 종료 후 적절한 기간 내 안전 삭제
   - 익명화/가명화 처리 후 장기 보존 가능

5. 윤리적 사용
   - IRB(Institutional Review Board) 승인 필수
   - 데이터는 연구 목적으로만 사용
   - 상업적 목적 사용 금지
===========================================
"""


def print_security_guidelines() -> None:
    """데이터 보안 가이드라인 출력"""
    print(DATA_SECURITY_GUIDELINES)
