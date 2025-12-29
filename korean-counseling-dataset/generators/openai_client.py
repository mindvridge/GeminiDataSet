"""
OpenAI 호환 API 클라이언트 모듈

Elice ML API 프록시를 통해 Gemini 3 Flash 모델에 접근합니다.
OpenAI Python 라이브러리를 사용하여 호환 API와 통신합니다.
Track B (일반상담) 데이터 생성에 사용됩니다.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from config.settings import Settings, get_settings

# OpenAI 라이브러리 임포트 시도
try:
    from openai import AsyncOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None
    OpenAI = None

logger = logging.getLogger(__name__)


@dataclass
class ResponseComponents:
    """
    API 응답 구성 요소

    GeminiClient와 호환되는 응답 형식을 유지합니다.
    """
    thought_trace: str = ""           # 사고 과정 (OpenAI API에서는 미지원)
    response_text: str = ""           # 최종 응답 텍스트
    raw_response: Any = None          # 원본 응답 객체
    usage_metadata: dict = field(default_factory=dict)  # 토큰 사용량


class OpenAICompatibleClient:
    """
    OpenAI 호환 API 클라이언트

    Elice ML API 프록시를 통해 Gemini 3 Flash 모델에 접근합니다.
    GeminiClient와 동일한 인터페이스를 제공하여 쉽게 교체 가능합니다.

    사용 예시:
    ```python
    client = OpenAICompatibleClient()
    response = await client.generate(
        system_prompt="당신은 임상심리전문가입니다...",
        user_prompt="상담 세션을 생성하세요."
    )
    print(response.response_text)
    ```
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ):
        """
        클라이언트 초기화

        Args:
            settings: 커스텀 설정 (기본값: get_settings())
        """
        self.settings = settings or get_settings()
        self.track = "B"  # OpenAI 클라이언트는 항상 Track B용

        # API 설정
        self.base_url = self.settings.elice_api_base_url
        self.api_key = self.settings.elice_api_key
        self.model_id = self.settings.elice_model
        self.temperature = self.settings.temperature_flash

        # 클라이언트 초기화
        self._client = None
        self._async_client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """OpenAI 클라이언트 초기화"""
        if not OPENAI_AVAILABLE:
            logger.warning(
                "openai 라이브러리가 설치되지 않았습니다. "
                "pip install openai 명령으로 설치하세요."
            )
            return

        if not self.base_url or not self.api_key:
            logger.warning(
                "Elice API 설정이 없습니다. "
                ".env 파일에 ELICE_API_BASE_URL과 ELICE_API_KEY를 설정하세요."
            )
            return

        try:
            # 동기 클라이언트
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

            # 비동기 클라이언트
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

            logger.info(
                f"OpenAI 호환 클라이언트 초기화 완료: "
                f"모델={self.model_id}, base_url={self.base_url}"
            )

        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self._client = None
            self._async_client = None

    def _extract_response_components(
        self,
        response: Any
    ) -> ResponseComponents:
        """
        OpenAI 응답에서 구성 요소 추출

        Args:
            response: OpenAI API 응답 객체

        Returns:
            ResponseComponents: 분리된 응답 구성 요소
        """
        components = ResponseComponents(raw_response=response)

        if response is None:
            return components

        try:
            # 응답 텍스트 추출
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    components.response_text = choice.message.content or ""

            # 사용량 메타데이터 추출
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                components.usage_metadata = {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                    "response_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                    "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                    "thoughts_tokens": 0,  # OpenAI API에서는 미지원
                }

        except Exception as e:
            logger.error(f"응답 파싱 오류: {e}")

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
            ResponseComponents: 생성된 응답
        """
        if not OPENAI_AVAILABLE or not self._async_client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return ResponseComponents()

        try:
            # 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # 대화 히스토리가 있는 경우
            if history:
                for msg in history:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            # 현재 사용자 프롬프트
            messages.append({
                "role": "user",
                "content": user_prompt
            })

            # API 호출
            response = await self._async_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.settings.max_output_tokens,
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
        if not OPENAI_AVAILABLE or not self._client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return ResponseComponents()

        try:
            # 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            if history:
                for msg in history:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            messages.append({
                "role": "user",
                "content": user_prompt
            })

            # API 호출
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.settings.max_output_tokens,
            )

            return self._extract_response_components(response)

        except Exception as e:
            logger.error(f"콘텐츠 생성 오류: {e}")
            return ResponseComponents()

    async def generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[list[dict]] = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
    ) -> ResponseComponents:
        """
        재시도 로직이 적용된 콘텐츠 생성

        지수 백오프: 5초 -> 15초 -> 45초 -> 120초 -> 300초

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            history: 대화 히스토리
            max_retries: 최대 재시도 횟수 (기본값: 5)
            retry_delay: 기본 재시도 대기 시간 (초, 기본값: 5.0)

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
                error_str = str(e).lower()

                # Rate Limit 에러 감지
                is_rate_limit = (
                    "429" in str(e) or
                    "rate" in error_str or
                    "limit" in error_str or
                    "quota" in error_str
                )

                if is_rate_limit:
                    logger.warning(
                        f"⚠️ Rate Limit 감지 (시도 {attempt + 1}/{max_retries + 1}): {e}"
                    )
                else:
                    logger.warning(
                        f"생성 실패 (시도 {attempt + 1}/{max_retries + 1}): {e}"
                    )

            if attempt < max_retries:
                # 지수 백오프 계산 (3배씩 증가, 최대 5분)
                wait_time = min(retry_delay * (3 ** attempt), 300)

                # Rate Limit 에러의 경우 추가 대기
                if last_error and ("429" in str(last_error) or "rate" in str(last_error).lower()):
                    wait_time = min(wait_time * 2, 300)
                    logger.info(f"⏳ Rate Limit 회복 대기 중... {wait_time:.0f}초")
                else:
                    logger.info(f"⏳ 재시도 대기 중... {wait_time:.0f}초")

                await asyncio.sleep(wait_time)

        logger.error(f"최대 재시도 횟수 초과: {last_error}")
        return ResponseComponents()

    def parse_json_response(
        self,
        response_text: str
    ) -> Optional[dict]:
        """
        JSON 응답 파싱

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

        Elice ML API는 별도 과금 정책이 있을 수 있습니다.
        여기서는 Gemini Flash 기준 가격으로 추정합니다.

        Args:
            prompt_tokens: 프롬프트 토큰 수
            response_tokens: 응답 토큰 수
            thoughts_tokens: 사고 과정 토큰 수 (미사용)

        Returns:
            추정 비용 (USD)
        """
        # Flash 모델 가격 (1M 토큰당 USD)
        input_price = 0.075
        output_price = 0.30

        input_cost = (prompt_tokens / 1_000_000) * input_price
        output_cost = (response_tokens / 1_000_000) * output_price

        return input_cost + output_cost

    def get_model_info(self) -> dict:
        """현재 모델 정보 반환"""
        return {
            "model_id": self.model_id,
            "track": self.track,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "api_type": "openai_compatible",
        }


# ==================== 편의 함수 ====================

def create_elice_client(settings: Optional[Settings] = None) -> OpenAICompatibleClient:
    """
    Elice ML API 클라이언트 생성 (Track B, 일반상담)

    OpenAI 호환 프록시를 통해 Gemini 3 Flash 모델에 접근합니다.
    """
    return OpenAICompatibleClient(settings=settings)
