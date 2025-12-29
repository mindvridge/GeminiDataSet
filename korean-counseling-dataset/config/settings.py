"""
환경 설정 모듈

Google Cloud 프로젝트 설정, API 키, 모델 설정 등을 관리합니다.
Pydantic Settings를 사용하여 .env 파일에서 자동으로 설정을 로드합니다.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    프로젝트 전역 설정

    환경 변수 또는 .env 파일에서 설정을 로드합니다.
    Vertex AI를 사용하여 Gemini 3 모델에 접근합니다.
    """

    # ==================== Google Cloud 설정 ====================
    google_cloud_project: str = Field(
        default="your-project-id",
        description="Google Cloud 프로젝트 ID"
    )
    google_cloud_location: str = Field(
        default="global",
        description="Vertex AI 리전 (Gemini 3 모델은 global 엔드포인트 필수)"
    )
    google_genai_use_vertexai: bool = Field(
        default=True,
        description="Vertex AI 사용 여부 (True: Vertex AI, False: AI Studio)"
    )
    google_api_key: str = Field(
        default="",
        description="Google AI Studio API 키 (Vertex AI 미사용 시 필요)"
    )

    # ==================== Gemini 모델 설정 ====================
    gemini_pro_model: str = Field(
        default="gemini-3-pro-preview",
        description="Gemini 3 Pro 모델 ID (고위험군 데이터 생성용, Thinking Mode 지원)"
    )
    gemini_flash_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini 3 Flash 모델 ID (일반상담 데이터 생성용)"
    )

    # ==================== 생성 설정 ====================
    max_output_tokens: int = Field(
        default=32768,
        description="최대 출력 토큰 수 (30-40턴 Long 세션용)"
    )
    temperature_pro: float = Field(
        default=0.7,
        description="Pro 모델 온도 (창의성 조절, 0.0~1.0)"
    )
    temperature_flash: float = Field(
        default=0.8,
        description="Flash 모델 온도"
    )
    top_p: float = Field(
        default=0.95,
        description="Top-p 샘플링 값"
    )
    top_k: int = Field(
        default=40,
        description="Top-k 샘플링 값"
    )

    # ==================== 배치 처리 설정 ====================
    batch_size: int = Field(
        default=10,
        description="배치당 처리할 데이터 수"
    )
    max_concurrent_requests: int = Field(
        default=5,
        description="최대 동시 요청 수"
    )
    retry_attempts: int = Field(
        default=5,
        description="API 호출 재시도 횟수 (429 에러 대응 강화)"
    )
    retry_delay: float = Field(
        default=5.0,
        description="재시도 기본 대기 시간 (초, 지수 백오프 적용: 5s->15s->45s->135s->300s)"
    )
    request_timeout: int = Field(
        default=120,
        description="API 요청 타임아웃 (초)"
    )
    max_budget_usd: float = Field(
        default=0.0,
        description="최대 예산 한도 (USD, 0=무제한)"
    )

    # ==================== 데이터 경로 설정 ====================
    data_base_dir: Path = Field(
        default=Path("data"),
        description="데이터 기본 디렉토리"
    )
    scenarios_dir: Path = Field(
        default=Path("data/scenarios"),
        description="시나리오 템플릿 디렉토리"
    )
    raw_data_dir: Path = Field(
        default=Path("data/raw"),
        description="생성된 원시 데이터 디렉토리"
    )
    validated_data_dir: Path = Field(
        default=Path("data/validated"),
        description="검증된 데이터 디렉토리"
    )
    final_data_dir: Path = Field(
        default=Path("data/final"),
        description="최종 데이터셋 디렉토리"
    )

    # ==================== 품질 검증 설정 ====================
    validation_sample_rate: float = Field(
        default=0.1,
        description="검증 샘플링 비율 (0.0~1.0)"
    )
    min_empathy_score: int = Field(
        default=3,
        description="최소 공감 점수 (1~5)"
    )
    min_korean_naturalness: int = Field(
        default=3,
        description="최소 한국어 자연스러움 점수 (1~5)"
    )

    # ==================== Elice ML API 설정 (Track B) ====================
    elice_api_base_url: str = Field(
        default="",
        description="Elice ML API Base URL (OpenAI 호환)"
    )
    elice_api_key: str = Field(
        default="",
        description="Elice ML API Key"
    )
    elice_model: str = Field(
        default="google/gemini-3-flash",
        description="Elice ML API 모델 ID"
    )
    track_b_api: Literal["elice", "vertex"] = Field(
        default="elice",
        description="Track B API 선택 (elice: Elice ML API, vertex: Vertex AI)"
    )

    # ==================== 로깅 설정 ====================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="로깅 레벨"
    )
    log_file: Path = Field(
        default=Path("logs/pipeline.log"),
        description="로그 파일 경로"
    )

    class Config:
        """Pydantic 설정"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    def get_project_root(self) -> Path:
        """프로젝트 루트 디렉토리 반환"""
        return Path(__file__).parent.parent

    def ensure_directories(self) -> None:
        """필요한 디렉토리들이 존재하는지 확인하고 없으면 생성"""
        directories = [
            self.data_base_dir,
            self.scenarios_dir,
            self.raw_data_dir,
            self.validated_data_dir,
            self.final_data_dir,
            self.log_file.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_model_config(self, track: Literal["A", "B"]) -> dict:
        """
        트랙에 따른 모델 설정 반환

        Args:
            track: "A" (고위험군, Pro) 또는 "B" (일반상담, Flash)

        Returns:
            모델 설정 딕셔너리
        """
        if track == "A":
            return {
                "model": self.gemini_pro_model,
                "temperature": self.temperature_pro,
                "thinking_level": "HIGH",
                "description": "고위험군 상담 (자살위기, 자해, 폭력 등)"
            }
        else:
            return {
                "model": self.gemini_flash_model,
                "temperature": self.temperature_flash,
                "thinking_level": "LOW",
                "description": "일반상담 (진로, 대인관계, 학업 등)"
            }


@lru_cache()
def get_settings() -> Settings:
    """
    설정 싱글톤 인스턴스 반환

    lru_cache를 사용하여 설정 객체를 캐싱합니다.
    애플리케이션 전체에서 동일한 설정 인스턴스를 공유합니다.

    Returns:
        Settings 인스턴스
    """
    settings = Settings()
    settings.ensure_directories()
    return settings


# 환경 변수 설정 (Vertex AI 사용을 위한)
def configure_environment() -> None:
    """
    Vertex AI 또는 AI Studio 사용을 위한 환경 변수 설정

    google-genai SDK가 올바른 인증 방식을 사용하도록 환경 변수를 설정합니다.
    """
    settings = get_settings()

    if settings.google_genai_use_vertexai:
        os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud_project
        os.environ["GOOGLE_CLOUD_LOCATION"] = settings.google_cloud_location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    else:
        # AI Studio 모드 - API 키 사용
        if settings.google_api_key:
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
