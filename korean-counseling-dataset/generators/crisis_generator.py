"""
고위험군 데이터 생성기 (Crisis Generator)

Track A 데이터 생성을 담당합니다:
- 자살 위기 (suicide_crisis)
- 자해 충동 (self_harm)
- 가정폭력 (domestic_violence)
- 성폭력 피해 (sexual_assault)
- 심각한 우울증 (severe_depression)

Gemini 3 Pro 모델을 사용하며, BLOCK_NONE 안전 설정과
HIGH Thinking Level을 적용합니다.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from config.settings import Settings, get_settings
from config.safety_config import get_crisis_safety_settings
from models.schemas import (
    CounselingCategory,
    CounselingSession,
    CounselingTurn,
    ClientUtterance,
    TherapistResponse,
    EmpathyTechnique,
    RiskLevel,
    GenerationRequest,
)
from models.prompts import (
    get_system_prompt,
    get_scenario_prompt,
    TRACK_A_CATEGORIES,
)

from .gemini_client import GeminiClient, ThinkingConfig, create_pro_client

logger = logging.getLogger(__name__)


class CrisisGenerator:
    """
    고위험군 상담 데이터 생성기

    자살, 자해, 폭력 피해 등 민감한 주제의 상담 데이터를 생성합니다.
    Gemini 3 Pro 모델의 깊은 추론 능력과 BLOCK_NONE 설정을 활용하여
    임상적으로 정확한 위기 개입 데이터를 생성합니다.

    사용 예시:
    ```python
    generator = CrisisGenerator()

    # 단일 세션 생성
    session = await generator.generate_session(
        category=CounselingCategory.SUICIDE_CRISIS,
        min_turns=5,
        max_turns=8
    )

    # 결과 저장
    generator.save_session(session, "data/raw/")
    ```
    """

    # 지원하는 카테고리 (Track A)
    SUPPORTED_CATEGORIES = TRACK_A_CATEGORIES

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[GeminiClient] = None,
    ):
        """
        생성기 초기화

        Args:
            settings: 설정 객체 (선택적)
            client: 커스텀 Gemini 클라이언트 (선택적)
        """
        self.settings = settings or get_settings()

        # Pro 모델 클라이언트 (Track A 전용)
        if client:
            self.client = client
        else:
            self.client = GeminiClient(
                track="A",
                settings=self.settings,
                custom_safety_config=get_crisis_safety_settings(),
                custom_thinking_config=ThinkingConfig(
                    include_thoughts=True,
                    thinking_level="HIGH"
                )
            )

        # 통계 추적
        self.stats = {
            "total_generated": 0,
            "total_failed": 0,
            "by_category": {},
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }

        logger.info("CrisisGenerator 초기화 완료 (Track A, Pro 모델)")

    def _validate_category(self, category: CounselingCategory) -> None:
        """카테고리 유효성 검사"""
        if category not in self.SUPPORTED_CATEGORIES:
            raise ValueError(
                f"지원하지 않는 카테고리: {category.value}. "
                f"Track A 카테고리만 지원합니다: "
                f"{[c.value for c in self.SUPPORTED_CATEGORIES]}"
            )

    async def generate_session(
        self,
        category: CounselingCategory,
        min_turns: int = 5,
        max_turns: int = 10,
        custom_scenario: Optional[str] = None,
    ) -> Optional[CounselingSession]:
        """
        단일 상담 세션 생성

        Args:
            category: 상담 카테고리 (Track A만 지원)
            min_turns: 최소 대화 턴 수
            max_turns: 최대 대화 턴 수
            custom_scenario: 커스텀 시나리오 (선택적)

        Returns:
            생성된 CounselingSession 또는 None (실패 시)
        """
        self._validate_category(category)

        # 시스템 프롬프트 구성
        system_prompt = get_system_prompt(track="A", include_thinking=True)

        # 시나리오 프롬프트 구성
        scenario_prompt = get_scenario_prompt(
            category=category,
            min_turns=min_turns,
            max_turns=max_turns,
            custom_scenario=custom_scenario
        )

        user_prompt = f"""
다음 시나리오에 맞는 심리상담 세션을 생성해주세요.

{scenario_prompt}

반드시 지정된 JSON 형식으로 출력하세요.
각 턴에서 상담사의 임상적 판단 과정(clinical_reasoning)을 상세히 기술하세요.
위기 상황에 대한 전문적인 개입이 포함되어야 합니다.
"""

        try:
            # API 호출
            response = await self.client.generate_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=self.settings.retry_attempts,
                retry_delay=self.settings.retry_delay,
            )

            # 응답 파싱
            parsed = self.client.parse_json_response(response.response_text)

            if not parsed:
                logger.error("JSON 파싱 실패")
                self.stats["total_failed"] += 1
                return None

            # CounselingSession 객체 생성
            session = self._parse_session(parsed, category, response.thought_trace)

            if session:
                # 통계 업데이트
                self.stats["total_generated"] += 1
                cat_key = category.value
                self.stats["by_category"][cat_key] = \
                    self.stats["by_category"].get(cat_key, 0) + 1

                if response.usage_metadata:
                    tokens = response.usage_metadata.get("total_tokens", 0)
                    self.stats["total_tokens"] += tokens
                    self.stats["estimated_cost"] += self.client.estimate_cost(
                        prompt_tokens=response.usage_metadata.get("prompt_tokens", 0),
                        response_tokens=response.usage_metadata.get("response_tokens", 0),
                        thoughts_tokens=response.usage_metadata.get("thoughts_tokens", 0),
                    )

                logger.info(
                    f"세션 생성 완료: {session.session_id} "
                    f"(카테고리: {category.value}, 턴 수: {len(session.turns)})"
                )

            return session

        except Exception as e:
            logger.error(f"세션 생성 오류: {e}")
            self.stats["total_failed"] += 1
            return None

    def _parse_session(
        self,
        data: dict,
        category: CounselingCategory,
        thought_trace: str = "",
    ) -> Optional[CounselingSession]:
        """
        파싱된 JSON을 CounselingSession 객체로 변환

        Args:
            data: 파싱된 JSON 딕셔너리
            category: 상담 카테고리
            thought_trace: Gemini의 사고 과정

        Returns:
            CounselingSession 객체 또는 None
        """
        try:
            # 대화 턴 파싱
            turns = []
            for turn_data in data.get("turns", []):
                client_data = turn_data.get("client", {})
                therapist_data = turn_data.get("therapist", {})

                # 공감 기법 파싱
                try:
                    empathy_tech = EmpathyTechnique(
                        therapist_data.get("empathy_technique", "반영하기")
                    )
                except ValueError:
                    empathy_tech = EmpathyTechnique.REFLECTION

                # 보조 기법 파싱
                secondary_techs = []
                for tech_name in therapist_data.get("secondary_techniques", []):
                    try:
                        secondary_techs.append(EmpathyTechnique(tech_name))
                    except ValueError:
                        continue

                turn = CounselingTurn(
                    turn_number=turn_data.get("turn_number", len(turns) + 1),
                    client=ClientUtterance(
                        text=client_data.get("text", ""),
                        emotional_state=client_data.get("emotional_state", ""),
                        risk_indicators=client_data.get("risk_indicators", []),
                        intensity=client_data.get("intensity", 5),
                    ),
                    therapist=TherapistResponse(
                        clinical_reasoning=therapist_data.get("clinical_reasoning", ""),
                        cognitive_distortions=therapist_data.get("cognitive_distortions", []),
                        intervention_strategy=therapist_data.get("intervention_strategy", ""),
                        utterance=therapist_data.get("utterance", ""),
                        empathy_technique=empathy_tech,
                        secondary_techniques=secondary_techs,
                        safety_action=therapist_data.get("safety_action"),
                    ),
                )
                turns.append(turn)

            # 위험 수준 파싱
            try:
                risk_level = RiskLevel(data.get("risk_level", "high"))
            except ValueError:
                risk_level = RiskLevel.HIGH

            # 세션 생성
            session = CounselingSession(
                session_id=data.get("session_id", str(uuid4())),
                category=category,
                risk_level=risk_level,
                client_profile=data.get("client_profile", ""),
                presenting_problem=data.get("presenting_problem", ""),
                turns=turns,
                session_summary=data.get("session_summary", ""),
                clinical_notes=data.get("clinical_notes", ""),
                treatment_goals=data.get("treatment_goals", []),
                homework=data.get("homework"),
                metadata={
                    "track": "A",
                    "model": self.client.model_id,
                    "thought_trace": thought_trace,
                    "generated_at": datetime.now().isoformat(),
                },
                model_used=self.client.model_id,
            )

            return session

        except Exception as e:
            logger.error(f"세션 파싱 오류: {e}")
            return None

    async def generate_batch(
        self,
        category: CounselingCategory,
        count: int = 10,
        min_turns: int = 5,
        max_turns: int = 10,
    ) -> list[CounselingSession]:
        """
        배치 세션 생성

        Args:
            category: 상담 카테고리
            count: 생성할 세션 수
            min_turns: 최소 대화 턴 수
            max_turns: 최대 대화 턴 수

        Returns:
            생성된 CounselingSession 리스트
        """
        self._validate_category(category)

        sessions = []
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)

        async def generate_with_semaphore(idx: int):
            async with semaphore:
                logger.info(f"세션 {idx + 1}/{count} 생성 중...")
                return await self.generate_session(
                    category=category,
                    min_turns=min_turns,
                    max_turns=max_turns,
                )

        # 병렬 생성
        tasks = [generate_with_semaphore(i) for i in range(count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, CounselingSession):
                sessions.append(result)
            elif isinstance(result, Exception):
                logger.error(f"세션 생성 중 예외 발생: {result}")

        logger.info(
            f"배치 생성 완료: {len(sessions)}/{count} 성공 "
            f"(카테고리: {category.value})"
        )

        return sessions

    def save_session(
        self,
        session: CounselingSession,
        output_dir: str | Path,
        format: str = "json",
    ) -> Path:
        """
        세션을 파일로 저장

        Args:
            session: 저장할 세션
            output_dir: 출력 디렉토리
            format: 저장 형식 ("json" 또는 "jsonl")

        Returns:
            저장된 파일 경로
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{session.category.value}_{session.session_id}.json"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                session.model_dump(mode="json"),
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"세션 저장 완료: {filepath}")
        return filepath

    def save_batch(
        self,
        sessions: list[CounselingSession],
        output_dir: str | Path,
        format: str = "jsonl",
    ) -> Path:
        """
        배치 세션을 파일로 저장

        Args:
            sessions: 저장할 세션 리스트
            output_dir: 출력 디렉토리
            format: 저장 형식 ("json" 또는 "jsonl")

        Returns:
            저장된 파일 경로
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crisis_batch_{timestamp}.{format}"
        filepath = output_dir / filename

        if format == "jsonl":
            with open(filepath, "w", encoding="utf-8") as f:
                for session in sessions:
                    json_line = json.dumps(
                        session.model_dump(mode="json"),
                        ensure_ascii=False
                    )
                    f.write(json_line + "\n")
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    [s.model_dump(mode="json") for s in sessions],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        logger.info(f"배치 저장 완료: {filepath} ({len(sessions)}건)")
        return filepath

    def get_stats(self) -> dict:
        """통계 반환"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["total_generated"] /
                (self.stats["total_generated"] + self.stats["total_failed"])
                if (self.stats["total_generated"] + self.stats["total_failed"]) > 0
                else 0.0
            ),
            "model_info": self.client.get_model_info(),
        }

    def reset_stats(self) -> None:
        """통계 초기화"""
        self.stats = {
            "total_generated": 0,
            "total_failed": 0,
            "by_category": {},
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }


# ==================== 편의 함수 ====================

async def generate_crisis_session(
    category: CounselingCategory,
    min_turns: int = 5,
    max_turns: int = 10,
) -> Optional[CounselingSession]:
    """
    단일 위기 상담 세션 생성 (편의 함수)

    Args:
        category: Track A 카테고리
        min_turns: 최소 턴 수
        max_turns: 최대 턴 수

    Returns:
        생성된 세션 또는 None
    """
    generator = CrisisGenerator()
    return await generator.generate_session(
        category=category,
        min_turns=min_turns,
        max_turns=max_turns,
    )
