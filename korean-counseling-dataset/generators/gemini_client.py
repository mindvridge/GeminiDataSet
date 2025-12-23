"""
Gemini API 클라이언트 모듈

google-genai SDK를 사용하여 Gemini 3 모델과 통신합니다.
Vertex AI를 통한 접근, Thinking Mode, 안전 설정 제어를 지원합니다.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from config.settings import Settings, get_settings
from config.safety_config import SafetyConfig, get_safety_settings

# google-genai SDK 임포트 시도
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

logger = logging.getLogger(__name__)


@dataclass
class ThinkingConfig:
    """
    Gemini 3 사고 모드 설정

    Gemini 3의 핵심 기능인 'Thinking Mode'를 제어합니다.
    사고 과정(Thought Trace)을 추출하여 XAI 데이터를 구축할 수 있습니다.
    """
    include_thoughts: bool = True
    thinking_level: Literal["NONE", "LOW", "MEDIUM", "HIGH"] = "HIGH"

    def to_genai_config(self) -> Any:
        """google-genai SDK 형식으로 변환"""
        if not GENAI_AVAILABLE:
            return None

        # ThinkingLevel enum 매핑
        level_mapping = {
            "NONE": None,  # 사고 모드 비활성화
            "LOW": types.ThinkingLevel.LOW,
            "MEDIUM": types.ThinkingLevel.MEDIUM,
            "HIGH": types.ThinkingLevel.HIGH,
        }

        if self.thinking_level == "NONE":
            return None

        return types.ThinkingConfig(
            include_thoughts=self.include_thoughts,
            thinking_level=level_mapping.get(self.thinking_level)
        )


@dataclass
class ResponseComponents:
    """
    Gemini 응답 구성 요소

    사고 과정(Thought Trace)과 최종 응답을 분리하여 저장합니다.
    """
    thought_trace: str = ""           # 사고 과정 텍스트
    response_text: str = ""           # 최종 응답 텍스트
    raw_response: Any = None          # 원본 응답 객체
    usage_metadata: dict = field(default_factory=dict)  # 토큰 사용량 등


class GeminiClient:
    """
    Gemini API 클라이언트

    Vertex AI를 통해 Gemini 3 모델에 접근하고,
    Thinking Mode와 안전 설정을 제어합니다.

    사용 예시:
    ```python
    client = GeminiClient(track="A")  # 고위험군용 Pro 모델
    response = await client.generate(
        system_prompt="당신은 임상심리전문가입니다...",
        user_prompt="자살 위기 상담 세션을 생성하세요."
    )
    print(response.thought_trace)  # 사고 과정
    print(response.response_text)   # 최종 응답
    ```
    """

    def __init__(
        self,
        track: Literal["A", "B"] = "B",
        settings: Optional[Settings] = None,
        custom_safety_config: Optional[SafetyConfig] = None,
        custom_thinking_config: Optional[ThinkingConfig] = None,
    ):
        """
        클라이언트 초기화

        Args:
            track: "A" (고위험군, Pro) 또는 "B" (일반상담, Flash)
            settings: 커스텀 설정 (기본값: get_settings())
            custom_safety_config: 커스텀 안전 설정
            custom_thinking_config: 커스텀 사고 모드 설정
        """
        self.track = track
        self.settings = settings or get_settings()
        self.safety_config = custom_safety_config or get_safety_settings(track)

        # 트랙에 따른 기본 사고 모드 설정
        if custom_thinking_config:
            self.thinking_config = custom_thinking_config
        else:
            self.thinking_config = ThinkingConfig(
                include_thoughts=True,
                thinking_level="HIGH" if track == "A" else "LOW"
            )

        # 모델 설정
        model_config = self.settings.get_model_config(track)
        self.model_id = model_config["model"]
        self.temperature = model_config["temperature"]

        # Gemini 클라이언트 초기화
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Gemini 클라이언트 초기화"""
        if not GENAI_AVAILABLE:
            logger.warning(
                "google-genai SDK가 설치되지 않았습니다. "
                "pip install google-genai 명령으로 설치하세요."
            )
            return

        # 환경 변수 설정
        os.environ["GOOGLE_CLOUD_PROJECT"] = self.settings.google_cloud_project
        os.environ["GOOGLE_CLOUD_LOCATION"] = self.settings.google_cloud_location

        if self.settings.google_genai_use_vertexai:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

        try:
            # Vertex AI 사용 시
            if self.settings.google_genai_use_vertexai:
                self._client = genai.Client(
                    vertexai=True,
                    project=self.settings.google_cloud_project,
                    location=self.settings.google_cloud_location,
                )
            else:
                # AI Studio 사용 시 (API 키 필요)
                self._client = genai.Client()

            logger.info(
                f"Gemini 클라이언트 초기화 완료: "
                f"모델={self.model_id}, 트랙={self.track}"
            )

        except Exception as e:
            logger.error(f"Gemini 클라이언트 초기화 실패: {e}")
            self._client = None

    def _build_generation_config(self) -> Any:
        """생성 설정 빌드"""
        if not GENAI_AVAILABLE:
            return None

        config_dict = {
            "temperature": self.temperature,
            "top_p": self.settings.top_p,
            "top_k": self.settings.top_k,
            "max_output_tokens": self.settings.max_output_tokens,
            "response_mime_type": "application/json",  # JSON 응답 강제
        }

        # Thinking Config 추가
        thinking_config = self.thinking_config.to_genai_config()
        if thinking_config:
            config_dict["thinking_config"] = thinking_config

        return types.GenerateContentConfig(**config_dict)

    def _extract_response_components(
        self,
        response: Any
    ) -> ResponseComponents:
        """
        Gemini 응답에서 사고 과정과 최종 응답 분리 추출

        Args:
            response: Gemini API 응답 객체

        Returns:
            ResponseComponents: 분리된 응답 구성 요소
        """
        components = ResponseComponents(raw_response=response)

        if response is None:
            return components

        try:
            # 응답에서 parts 추출
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        # 사고 과정(Thought) 추출
                        # SDK 버전에 따라 part.thought가 bool이거나 별도 속성일 수 있음
                        if getattr(part, 'thought', False):
                            components.thought_trace += part.text + "\n"
                        else:
                            components.response_text += part.text

            # 사용량 메타데이터 추출
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                components.usage_metadata = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "response_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0),
                    "thoughts_tokens": getattr(usage, 'thoughts_token_count', 0),
                }

        except Exception as e:
            logger.error(f"응답 파싱 오류: {e}")
            # 폴백: 전체 텍스트를 응답으로 처리
            if hasattr(response, 'text'):
                components.response_text = response.text

        return components

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict]] = None,
    ) -> ResponseComponents:
        """
        비동기 콘텐츠 생성

        Args:
            system_prompt: 시스템 프롬프트 (상담사 페르소나 정의)
            user_prompt: 사용자 프롬프트 (생성 지시)
            history: 대화 히스토리 (선택적)

        Returns:
            ResponseComponents: 생성된 응답 (사고 과정 + 최종 응답)
        """
        if not GENAI_AVAILABLE or not self._client:
            logger.error("Gemini 클라이언트가 초기화되지 않았습니다.")
            return ResponseComponents()

        try:
            # 콘텐츠 구성
            contents = []

            # 대화 히스토리가 있는 경우
            if history:
                for msg in history:
                    contents.append(
                        types.Content(
                            role=msg.get("role", "user"),
                            parts=[types.Part.from_text(msg.get("content", ""))]
                        )
                    )

            # 현재 사용자 프롬프트
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(user_prompt)]
                )
            )

            # 생성 설정
            generation_config = self._build_generation_config()

            # 안전 설정
            safety_settings = self.safety_config.to_genai_settings()

            # API 호출
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    top_p=self.settings.top_p,
                    top_k=self.settings.top_k,
                    max_output_tokens=self.settings.max_output_tokens,
                    safety_settings=safety_settings,
                    thinking_config=self.thinking_config.to_genai_config(),
                ),
            )

            return self._extract_response_components(response)

        except Exception as e:
            logger.error(f"콘텐츠 생성 오류: {e}")
            return ResponseComponents()

    def generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict]] = None,
    ) -> ResponseComponents:
        """
        동기 콘텐츠 생성

        비동기 환경이 아닌 경우 사용합니다.
        """
        return asyncio.run(self.generate(system_prompt, user_prompt, history))

    async def generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict]] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> ResponseComponents:
        """
        재시도 로직이 포함된 콘텐츠 생성

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            history: 대화 히스토리
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간 대기 시간 (초)

        Returns:
            ResponseComponents: 생성된 응답
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.generate(system_prompt, user_prompt, history)

                # 응답 유효성 검사
                if response.response_text.strip():
                    return response

                logger.warning(f"빈 응답 수신 (시도 {attempt + 1}/{max_retries + 1})")

            except Exception as e:
                last_error = e
                logger.warning(
                    f"생성 실패 (시도 {attempt + 1}/{max_retries + 1}): {e}"
                )

            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (attempt + 1))  # 지수 백오프

        logger.error(f"최대 재시도 횟수 초과: {last_error}")
        return ResponseComponents()

    def parse_json_response(
        self,
        response_text: str
    ) -> Optional[dict]:
        """
        JSON 응답 파싱

        Gemini가 생성한 JSON 문자열을 파이썬 딕셔너리로 변환합니다.
        마크다운 코드 블록도 처리합니다.

        Args:
            response_text: JSON 문자열

        Returns:
            파싱된 딕셔너리 또는 None
        """
        if not response_text:
            return None

        try:
            # 마크다운 코드 블록 제거
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            return json.loads(text.strip())

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            logger.debug(f"원본 텍스트: {response_text[:500]}...")
            return None

    def estimate_cost(
        self,
        prompt_tokens: int,
        response_tokens: int,
        thoughts_tokens: int = 0
    ) -> float:
        """
        비용 추정 (USD)

        2024년 기준 Gemini 가격을 기반으로 추정합니다.
        실제 가격은 Google Cloud 가격 정책을 확인하세요.

        Args:
            prompt_tokens: 프롬프트 토큰 수
            response_tokens: 응답 토큰 수
            thoughts_tokens: 사고 과정 토큰 수

        Returns:
            추정 비용 (USD)
        """
        # 가격 (1M 토큰당 USD) - 예시 가격
        if self.track == "A":  # Pro
            input_price = 1.25
            output_price = 5.00
        else:  # Flash
            input_price = 0.075
            output_price = 0.30

        input_cost = (prompt_tokens / 1_000_000) * input_price
        output_cost = ((response_tokens + thoughts_tokens) / 1_000_000) * output_price

        return input_cost + output_cost

    def get_model_info(self) -> dict:
        """현재 모델 정보 반환"""
        return {
            "model_id": self.model_id,
            "track": self.track,
            "temperature": self.temperature,
            "thinking_level": self.thinking_config.thinking_level,
            "safety_config": self.safety_config.description,
            "vertex_ai": self.settings.google_genai_use_vertexai,
        }


# ==================== 편의 함수 ====================

def create_pro_client(settings: Optional[Settings] = None) -> GeminiClient:
    """
    Pro 모델 클라이언트 생성 (Track A, 고위험군)

    고위험 상담 데이터 생성용으로 최적화된 클라이언트입니다.
    BLOCK_NONE 설정과 HIGH Thinking Level을 사용합니다.
    """
    return GeminiClient(
        track="A",
        settings=settings,
        custom_thinking_config=ThinkingConfig(
            include_thoughts=True,
            thinking_level="HIGH"
        )
    )


def create_flash_client(settings: Optional[Settings] = None) -> GeminiClient:
    """
    Flash 모델 클라이언트 생성 (Track B, 일반상담)

    일반 상담 데이터 대량 생성용으로 최적화된 클라이언트입니다.
    BLOCK_ONLY_HIGH 설정과 LOW Thinking Level을 사용합니다.
    """
    return GeminiClient(
        track="B",
        settings=settings,
        custom_thinking_config=ThinkingConfig(
            include_thoughts=True,
            thinking_level="LOW"
        )
    )
