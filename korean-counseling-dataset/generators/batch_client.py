"""
Gemini Batch API 클라이언트

대량 데이터 생성을 위한 Batch API를 사용합니다.
Standard API 대비 50% 비용 절감 효과가 있습니다.

사용법:
    client = BatchClient(track="A")
    job = await client.create_batch_job(requests)
    results = await client.wait_for_completion(job)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

from config.settings import Settings, get_settings
from config.safety_config import get_crisis_safety_settings, get_general_safety_settings
from models.schemas import CounselingCategory

logger = logging.getLogger(__name__)

# google-genai SDK 임포트
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None


@dataclass
class BatchJobStatus:
    """배치 작업 상태"""
    job_id: str
    job_name: str
    state: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    error_message: Optional[str] = None


@dataclass
class BatchResult:
    """배치 결과"""
    job_id: str
    responses: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    estimated_cost: float = 0.0


class BatchClient:
    """
    Gemini Batch API 클라이언트

    대량 데이터 생성 시 50% 비용 절감을 위해 Batch API를 사용합니다.

    가격 비교 (1M 토큰당):
    - Standard: 입력 $2.00, 출력 $12.00
    - Batch:    입력 $1.00, 출력 $6.00

    사용 예시:
    ```python
    client = BatchClient(track="A")

    # 요청 생성
    requests = [
        {"system": "...", "user": "..."},
        {"system": "...", "user": "..."},
    ]

    # 배치 작업 제출
    job = await client.create_batch_job(requests, display_name="pro-seed-batch")

    # 완료 대기 (폴링)
    result = await client.wait_for_completion(job, poll_interval=60)

    # 결과 처리
    for response in result.responses:
        print(response)
    ```
    """

    # 완료 상태 목록
    COMPLETED_STATES = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }

    def __init__(
        self,
        track: Literal["A", "B"] = "A",
        settings: Optional[Settings] = None,
    ):
        """
        클라이언트 초기화

        Args:
            track: "A" (고위험군, Pro) 또는 "B" (일반상담, Flash)
            settings: 설정 객체
        """
        self.track = track
        self.settings = settings or get_settings()

        # 모델 설정
        model_config = self.settings.get_model_config(track)
        self.model_id = model_config["model"]
        self.temperature = model_config["temperature"]

        # 클라이언트 초기화
        self._client = None
        self._initialize_client()

        # 통계
        self.stats = {
            "total_jobs": 0,
            "total_requests": 0,
            "total_responses": 0,
            "total_errors": 0,
            "estimated_cost": 0.0,
        }

        # 안전 설정 로그
        if track == "A":
            logger.info(f"BatchClient 초기화 완료: 모델={self.model_id}, 트랙={track}, 안전설정=BLOCK_NONE")
        else:
            logger.info(f"BatchClient 초기화 완료: 모델={self.model_id}, 트랙={track}, 안전설정=기본")

    def _initialize_client(self) -> None:
        """Gemini 클라이언트 초기화"""
        if not GENAI_AVAILABLE:
            logger.warning("google-genai SDK가 설치되지 않았습니다.")
            return

        try:
            if self.settings.google_genai_use_vertexai:
                self._client = genai.Client(
                    vertexai=True,
                    project=self.settings.google_cloud_project,
                    location=self.settings.google_cloud_location,
                )
            else:
                self._client = genai.Client()

            logger.info("Batch 클라이언트 초기화 완료")

        except Exception as e:
            logger.error(f"Batch 클라이언트 초기화 실패: {e}")
            self._client = None

    def _get_safety_settings(self) -> list[dict]:
        """트랙에 따른 안전 설정 반환"""
        if self.track == "A":
            # 고위험군: BLOCK_NONE 설정
            safety_config = get_crisis_safety_settings()
        else:
            # 일반상담: 기본 설정
            safety_config = get_general_safety_settings()

        # Batch API 형식으로 변환
        safety_settings = []
        for setting in safety_config:
            safety_settings.append({
                'category': setting.category.name if hasattr(setting.category, 'name') else str(setting.category),
                'threshold': setting.threshold.name if hasattr(setting.threshold, 'name') else str(setting.threshold),
            })
        return safety_settings

    def _build_request(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """단일 요청 빌드"""
        return {
            'contents': [
                {
                    'parts': [{'text': system_prompt}],
                    'role': 'user'
                },
                {
                    'parts': [{'text': user_prompt}],
                    'role': 'user'
                }
            ],
            'generationConfig': {
                'temperature': self.temperature,
                'topP': self.settings.top_p,
                'topK': self.settings.top_k,
                'maxOutputTokens': self.settings.max_output_tokens,
            },
            'safetySettings': self._get_safety_settings(),
        }

    def build_requests(
        self,
        prompts: list[dict],
    ) -> list[dict]:
        """
        여러 프롬프트를 배치 요청 형식으로 변환

        Args:
            prompts: [{"system": "...", "user": "..."}, ...]

        Returns:
            배치 요청 리스트
        """
        requests = []
        for prompt in prompts:
            req = self._build_request(
                system_prompt=prompt.get("system", ""),
                user_prompt=prompt.get("user", ""),
            )
            requests.append(req)
        return requests

    async def create_batch_job(
        self,
        requests: list[dict],
        display_name: Optional[str] = None,
    ) -> BatchJobStatus:
        """
        배치 작업 생성 및 제출

        Args:
            requests: 배치 요청 리스트 (build_requests()로 생성)
            display_name: 작업 표시 이름

        Returns:
            BatchJobStatus 객체
        """
        if not self._client:
            raise RuntimeError("Batch 클라이언트가 초기화되지 않았습니다.")

        job_id = str(uuid4())[:8]
        display_name = display_name or f"batch-{self.track}-{job_id}"

        try:
            # 배치 작업 생성
            batch_job = self._client.batches.create(
                model=f"models/{self.model_id}",
                src=requests,
                config={'display_name': display_name},
            )

            status = BatchJobStatus(
                job_id=job_id,
                job_name=batch_job.name,
                state=batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state),
                created_at=datetime.now(),
                total_requests=len(requests),
            )

            self.stats["total_jobs"] += 1
            self.stats["total_requests"] += len(requests)

            logger.info(f"배치 작업 생성 완료: {status.job_name} ({len(requests)}건)")

            return status

        except Exception as e:
            logger.error(f"배치 작업 생성 실패: {e}")
            raise

    async def get_job_status(self, job_name: str) -> BatchJobStatus:
        """작업 상태 조회"""
        if not self._client:
            raise RuntimeError("Batch 클라이언트가 초기화되지 않았습니다.")

        batch_job = self._client.batches.get(name=job_name)

        state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)

        return BatchJobStatus(
            job_id=job_name.split('/')[-1],
            job_name=job_name,
            state=state_name,
            created_at=datetime.now(),
            completed_at=datetime.now() if state_name in self.COMPLETED_STATES else None,
        )

    async def wait_for_completion(
        self,
        job_status: BatchJobStatus,
        poll_interval: int = 60,
        max_wait_hours: int = 24,
        progress_callback: Optional[callable] = None,
    ) -> BatchResult:
        """
        배치 작업 완료 대기

        Args:
            job_status: 작업 상태
            poll_interval: 상태 확인 간격 (초)
            max_wait_hours: 최대 대기 시간 (시간)
            progress_callback: 진행률 콜백 함수

        Returns:
            BatchResult 객체
        """
        if not self._client:
            raise RuntimeError("Batch 클라이언트가 초기화되지 않았습니다.")

        job_name = job_status.job_name
        start_time = time.time()
        max_wait_seconds = max_wait_hours * 3600

        logger.info(f"배치 작업 완료 대기 중: {job_name}")

        while True:
            # 상태 확인
            batch_job = self._client.batches.get(name=job_name)
            state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)

            if progress_callback:
                progress_callback(state_name)

            # 완료 확인
            if state_name in self.COMPLETED_STATES:
                logger.info(f"배치 작업 완료: {state_name}")
                break

            # 타임아웃 확인
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                logger.warning(f"배치 작업 타임아웃: {max_wait_hours}시간 초과")
                break

            # 대기
            logger.info(f"배치 상태: {state_name} (경과: {elapsed/60:.1f}분)")
            await asyncio.sleep(poll_interval)

        # 결과 추출
        return self._extract_results(batch_job)

    def _extract_results(self, batch_job: Any) -> BatchResult:
        """배치 결과 추출"""
        result = BatchResult(
            job_id=batch_job.name.split('/')[-1] if hasattr(batch_job, 'name') else "",
        )

        try:
            # 인라인 응답 처리
            if hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'inlined_responses'):
                for inline_response in batch_job.dest.inlined_responses:
                    if hasattr(inline_response, 'response') and inline_response.response:
                        # 텍스트 추출
                        text = ""
                        if hasattr(inline_response.response, 'text'):
                            text = inline_response.response.text
                        elif hasattr(inline_response.response, 'candidates'):
                            for candidate in inline_response.response.candidates:
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text'):
                                            text += part.text

                        result.responses.append({
                            "text": text,
                            "raw": inline_response.response,
                        })
                        self.stats["total_responses"] += 1

                    elif hasattr(inline_response, 'error') and inline_response.error:
                        result.errors.append({
                            "error": str(inline_response.error),
                        })
                        self.stats["total_errors"] += 1

            # 비용 추정 (배치 가격) - 실제 출력 기반
            # 입력: $1.00/1M, 출력: $6.00/1M
            estimated_input_tokens = len(result.responses) * 700  # 시스템 + 사용자 프롬프트
            # 실제 출력 토큰 계산 (응답 텍스트 길이 기반, 한글 1자 ≈ 1.5 토큰)
            estimated_output_tokens = 0
            for resp in result.responses:
                text = resp.get("text", "")
                # 한글 기준 토큰 추정 (대략 글자수 * 1.5)
                estimated_output_tokens += int(len(text) * 1.5)

            result.total_tokens = estimated_input_tokens + estimated_output_tokens
            result.estimated_cost = (
                (estimated_input_tokens / 1_000_000) * 1.00 +
                (estimated_output_tokens / 1_000_000) * 6.00
            )

            self.stats["estimated_cost"] += result.estimated_cost

        except Exception as e:
            logger.error(f"결과 추출 오류: {e}")

        return result

    async def cancel_job(self, job_name: str) -> bool:
        """배치 작업 취소"""
        if not self._client:
            return False

        try:
            self._client.batches.cancel(name=job_name)
            logger.info(f"배치 작업 취소됨: {job_name}")
            return True
        except Exception as e:
            logger.error(f"배치 작업 취소 실패: {e}")
            return False

    def get_stats(self) -> dict:
        """통계 반환"""
        return {
            **self.stats,
            "model": self.model_id,
            "track": self.track,
        }


# ==================== 편의 함수 ====================

async def run_batch_generation(
    prompts: list[dict],
    track: Literal["A", "B"] = "A",
    display_name: Optional[str] = None,
    poll_interval: int = 60,
) -> BatchResult:
    """
    배치 생성 실행 (편의 함수)

    Args:
        prompts: [{"system": "...", "user": "..."}, ...]
        track: "A" 또는 "B"
        display_name: 작업 이름
        poll_interval: 폴링 간격 (초)

    Returns:
        BatchResult
    """
    client = BatchClient(track=track)
    requests = client.build_requests(prompts)

    job = await client.create_batch_job(requests, display_name=display_name)
    result = await client.wait_for_completion(job, poll_interval=poll_interval)

    return result
