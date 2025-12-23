"""
일반상담 데이터 생성기 (General Generator)

Track B 데이터 생성을 담당합니다:
- 진로 고민 (career)
- 대인관계 (relationship)
- 학업 스트레스 (academic)
- 경미한 불안 (mild_anxiety)
- 가족 갈등 (family_conflict)
- 직장 스트레스 (workplace)

Gemini 3 Flash 모델을 사용하며, BLOCK_ONLY_HIGH 안전 설정과
LOW Thinking Level을 적용하여 비용 효율적인 대량 생산을 지원합니다.

Pro 모델에서 생성된 고품질 예시를 Few-shot으로 활용하여
Flash 모델의 품질을 향상시키는 Teacher-Student 전략을 구현합니다.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from config.settings import Settings, get_settings
from config.safety_config import get_general_safety_settings
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
    get_few_shot_examples,
    format_session_as_example,
    TRACK_B_CATEGORIES,
)

from .gemini_client import GeminiClient, ThinkingConfig, create_flash_client

logger = logging.getLogger(__name__)


class GeneralGenerator:
    """
    일반상담 데이터 생성기

    진로, 대인관계, 학업 등 일반적인 상담 주제의 데이터를 대량 생성합니다.
    Gemini 3 Flash 모델을 사용하여 비용을 최소화하면서도
    Pro 모델의 Few-shot 예시를 활용해 품질을 유지합니다.

    Teacher-Student 증류(Distillation) 전략:
    1. Pro 모델(Teacher)이 생성한 고품질 예시를 저장
    2. Flash 모델(Student)의 프롬프트에 예시 삽입
    3. Flash가 Pro의 패턴을 모방하여 생성

    사용 예시:
    ```python
    generator = GeneralGenerator()

    # Few-shot 예시 로드 (Pro에서 생성된 것)
    generator.load_few_shot_examples("data/examples/")

    # 배치 생성
    sessions = await generator.generate_batch(
        category=CounselingCategory.CAREER,
        count=100
    )
    ```
    """

    # 지원하는 카테고리 (Track B)
    SUPPORTED_CATEGORIES = TRACK_B_CATEGORIES

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[GeminiClient] = None,
        few_shot_examples: Optional[dict[CounselingCategory, list[CounselingSession]]] = None,
    ):
        """
        생성기 초기화

        Args:
            settings: 설정 객체 (선택적)
            client: 커스텀 Gemini 클라이언트 (선택적)
            few_shot_examples: 카테고리별 Few-shot 예시 (선택적)
        """
        self.settings = settings or get_settings()

        # Flash 모델 클라이언트 (Track B 전용)
        if client:
            self.client = client
        else:
            self.client = GeminiClient(
                track="B",
                settings=self.settings,
                custom_safety_config=get_general_safety_settings(),
                custom_thinking_config=ThinkingConfig(
                    include_thoughts=True,
                    thinking_level="LOW"
                )
            )

        # Few-shot 예시 저장소
        self.few_shot_examples: dict[CounselingCategory, list[CounselingSession]] = \
            few_shot_examples or {}

        # 통계 추적
        self.stats = {
            "total_generated": 0,
            "total_failed": 0,
            "by_category": {},
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }

        logger.info("GeneralGenerator 초기화 완료 (Track B, Flash 모델)")

    def _validate_category(self, category: CounselingCategory) -> None:
        """카테고리 유효성 검사"""
        if category not in self.SUPPORTED_CATEGORIES:
            raise ValueError(
                f"지원하지 않는 카테고리: {category.value}. "
                f"Track B 카테고리만 지원합니다: "
                f"{[c.value for c in self.SUPPORTED_CATEGORIES]}"
            )

    def add_few_shot_example(
        self,
        category: CounselingCategory,
        session: CounselingSession,
    ) -> None:
        """
        Few-shot 예시 추가

        Pro 모델에서 생성된 고품질 세션을 예시로 추가합니다.

        Args:
            category: 상담 카테고리
            session: 예시로 사용할 세션
        """
        if category not in self.few_shot_examples:
            self.few_shot_examples[category] = []

        self.few_shot_examples[category].append(session)
        logger.info(
            f"Few-shot 예시 추가: {category.value} "
            f"(총 {len(self.few_shot_examples[category])}개)"
        )

    def load_few_shot_examples(
        self,
        examples_dir: str | Path,
        max_per_category: int = 3,
    ) -> None:
        """
        파일에서 Few-shot 예시 로드

        Args:
            examples_dir: 예시 파일이 있는 디렉토리
            max_per_category: 카테고리당 최대 예시 수
        """
        examples_dir = Path(examples_dir)

        if not examples_dir.exists():
            logger.warning(f"예시 디렉토리가 없습니다: {examples_dir}")
            return

        for category in self.SUPPORTED_CATEGORIES:
            category_files = list(examples_dir.glob(f"{category.value}_*.json"))

            for file_path in category_files[:max_per_category]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    session = CounselingSession(**data)
                    self.add_few_shot_example(category, session)

                except Exception as e:
                    logger.error(f"예시 로드 실패 ({file_path}): {e}")

        logger.info(f"Few-shot 예시 로드 완료: {sum(len(v) for v in self.few_shot_examples.values())}개")

    def _build_few_shot_prompt(
        self,
        category: CounselingCategory,
        max_examples: int = 2,
    ) -> str:
        """Few-shot 프롬프트 구성"""
        examples = self.few_shot_examples.get(category, [])

        if not examples:
            # 내장 예시 사용
            return "\n".join(get_few_shot_examples(category, max_examples))

        # 저장된 고품질 예시 사용
        example_texts = []
        for i, session in enumerate(examples[:max_examples], 1):
            example_text = format_session_as_example(session)
            example_texts.append(f"### 참고 예시 {i}\n{example_text}")

        return "\n\n".join(example_texts)

    async def generate_session(
        self,
        category: CounselingCategory,
        min_turns: int = 5,
        max_turns: int = 10,
        custom_scenario: Optional[str] = None,
        use_few_shot: bool = True,
    ) -> Optional[CounselingSession]:
        """
        단일 상담 세션 생성

        Args:
            category: 상담 카테고리 (Track B만 지원)
            min_turns: 최소 대화 턴 수
            max_turns: 최대 대화 턴 수
            custom_scenario: 커스텀 시나리오 (선택적)
            use_few_shot: Few-shot 예시 사용 여부

        Returns:
            생성된 CounselingSession 또는 None (실패 시)
        """
        self._validate_category(category)

        # 시스템 프롬프트 구성
        system_prompt = get_system_prompt(track="B", include_thinking=True)

        # 시나리오 프롬프트 구성
        scenario_prompt = get_scenario_prompt(
            category=category,
            min_turns=min_turns,
            max_turns=max_turns,
            custom_scenario=custom_scenario
        )

        # Few-shot 프롬프트
        few_shot_prompt = ""
        if use_few_shot:
            few_shot_prompt = self._build_few_shot_prompt(category)

        user_prompt = f"""
다음 시나리오에 맞는 심리상담 세션을 생성해주세요.

{scenario_prompt}

{f"## 참고 예시 (이 패턴을 따라 생성하세요){chr(10)}{few_shot_prompt}" if few_shot_prompt else ""}

반드시 지정된 JSON 형식으로 출력하세요.
자연스러운 한국어 구어체를 사용하고, 공감적 화법을 유지하세요.
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
                    tokens = response.usage_metadata.get("total_tokens", 0) or 0
                    self.stats["total_tokens"] += tokens
                    self.stats["estimated_cost"] += self.client.estimate_cost(
                        prompt_tokens=response.usage_metadata.get("prompt_tokens", 0) or 0,
                        response_tokens=response.usage_metadata.get("response_tokens", 0) or 0,
                        thoughts_tokens=response.usage_metadata.get("thoughts_tokens", 0) or 0,
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

            # 위험 수준 파싱 (Track B는 보통 moderate 또는 low)
            try:
                risk_level = RiskLevel(data.get("risk_level", "moderate"))
            except ValueError:
                risk_level = RiskLevel.MODERATE

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
                    "track": "B",
                    "model": self.client.model_id,
                    "thought_trace": thought_trace,
                    "used_few_shot": len(self.few_shot_examples.get(category, [])) > 0,
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
        use_few_shot: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> list[CounselingSession]:
        """
        배치 세션 생성

        Args:
            category: 상담 카테고리
            count: 생성할 세션 수
            min_turns: 최소 대화 턴 수
            max_turns: 최대 대화 턴 수
            use_few_shot: Few-shot 예시 사용 여부
            progress_callback: 진행률 콜백 함수

        Returns:
            생성된 CounselingSession 리스트
        """
        self._validate_category(category)

        sessions = []
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)
        completed = 0

        async def generate_with_semaphore(idx: int):
            nonlocal completed
            async with semaphore:
                result = await self.generate_session(
                    category=category,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    use_few_shot=use_few_shot,
                )
                completed += 1

                if progress_callback:
                    progress_callback(completed, count)
                else:
                    logger.info(f"진행률: {completed}/{count} ({completed/count*100:.1f}%)")

                return result

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

    async def generate_all_categories(
        self,
        count_per_category: int = 10,
        min_turns: int = 5,
        max_turns: int = 10,
    ) -> dict[CounselingCategory, list[CounselingSession]]:
        """
        모든 Track B 카테고리에 대해 데이터 생성

        Args:
            count_per_category: 카테고리당 생성 수
            min_turns: 최소 턴 수
            max_turns: 최대 턴 수

        Returns:
            카테고리별 세션 딕셔너리
        """
        all_sessions = {}

        for category in self.SUPPORTED_CATEGORIES:
            logger.info(f"카테고리 생성 시작: {category.value}")

            sessions = await self.generate_batch(
                category=category,
                count=count_per_category,
                min_turns=min_turns,
                max_turns=max_turns,
            )

            all_sessions[category] = sessions

        total = sum(len(s) for s in all_sessions.values())
        logger.info(f"전체 생성 완료: {total}개 세션")

        return all_sessions

    def save_session(
        self,
        session: CounselingSession,
        output_dir: str | Path,
    ) -> Path:
        """세션을 파일로 저장"""
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

        return filepath

    def save_batch(
        self,
        sessions: list[CounselingSession],
        output_dir: str | Path,
        format: str = "jsonl",
    ) -> Path:
        """배치 세션을 파일로 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"general_batch_{timestamp}.{format}"
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
            "few_shot_examples_count": sum(
                len(v) for v in self.few_shot_examples.values()
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

async def generate_general_session(
    category: CounselingCategory,
    min_turns: int = 5,
    max_turns: int = 10,
) -> Optional[CounselingSession]:
    """
    단일 일반 상담 세션 생성 (편의 함수)

    Args:
        category: Track B 카테고리
        min_turns: 최소 턴 수
        max_turns: 최대 턴 수

    Returns:
        생성된 세션 또는 None
    """
    generator = GeneralGenerator()
    return await generator.generate_session(
        category=category,
        min_turns=min_turns,
        max_turns=max_turns,
    )
