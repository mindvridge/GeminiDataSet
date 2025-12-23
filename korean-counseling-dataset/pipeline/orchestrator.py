"""
파이프라인 오케스트레이터 (Pipeline Orchestrator)

전체 데이터셋 구축 파이프라인을 조율합니다:
1. 고위험군 데이터 생성 (Pro 모델)
2. Pro 예시를 Few-shot으로 활용하여 일반 데이터 생성 (Flash 모델)
3. 품질 검증 및 필터링
4. 최종 데이터셋 구축
"""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from config.settings import Settings, get_settings
from models.schemas import (
    CounselingCategory,
    CounselingSession,
    BatchResult,
    QualityScore,
)
from models.prompts import TRACK_A_CATEGORIES, TRACK_B_CATEGORIES
from generators.crisis_generator import CrisisGenerator
from generators.general_generator import GeneralGenerator
from validators.quality_checker import QualityChecker, EvaluationCriteria

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    파이프라인 설정

    전체 데이터셋 구축 파이프라인의 설정을 정의합니다.
    """
    # 생성 설정
    track_a_count: int = 50           # Track A 카테고리당 생성 수
    track_b_count: int = 200          # Track B 카테고리당 생성 수
    min_turns: int = 5                # 최소 대화 턴 수
    max_turns: int = 10               # 최대 대화 턴 수

    # Few-shot 설정
    few_shot_examples_per_category: int = 3  # 카테고리당 Few-shot 예시 수

    # 검증 설정
    validate: bool = True             # 품질 검증 실행 여부
    validation_sample_rate: float = 0.2  # 검증 샘플링 비율
    min_quality_threshold: float = 0.6   # 최소 품질 임계값

    # 출력 설정
    output_format: Literal["json", "jsonl", "both"] = "both"
    include_thought_trace: bool = True  # 사고 과정 포함 여부

    # 후처리 설정
    mask_sensitive_info: bool = True  # 민감정보 마스킹
    deduplicate: bool = True          # 중복 제거

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "track_a_count": self.track_a_count,
            "track_b_count": self.track_b_count,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "few_shot_examples_per_category": self.few_shot_examples_per_category,
            "validate": self.validate,
            "validation_sample_rate": self.validation_sample_rate,
            "min_quality_threshold": self.min_quality_threshold,
            "output_format": self.output_format,
            "include_thought_trace": self.include_thought_trace,
            "mask_sensitive_info": self.mask_sensitive_info,
            "deduplicate": self.deduplicate,
        }


@dataclass
class PipelineStats:
    """파이프라인 실행 통계"""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Track A
    track_a_generated: int = 0
    track_a_validated: int = 0
    track_a_passed: int = 0

    # Track B
    track_b_generated: int = 0
    track_b_validated: int = 0
    track_b_passed: int = 0

    # 최종
    total_final: int = 0
    total_cost_usd: float = 0.0

    # 카테고리별
    by_category: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else None
            ),
            "track_a": {
                "generated": self.track_a_generated,
                "validated": self.track_a_validated,
                "passed": self.track_a_passed,
            },
            "track_b": {
                "generated": self.track_b_generated,
                "validated": self.track_b_validated,
                "passed": self.track_b_passed,
            },
            "total_final": self.total_final,
            "total_cost_usd": self.total_cost_usd,
            "by_category": self.by_category,
        }


class PipelineOrchestrator:
    """
    파이프라인 오케스트레이터

    전체 데이터셋 구축 프로세스를 조율합니다.
    하이브리드 생산 전략(Pro Seed + Flash Scale)을 구현합니다.

    사용 예시:
    ```python
    config = PipelineConfig(
        track_a_count=50,
        track_b_count=200,
        validate=True,
    )

    orchestrator = PipelineOrchestrator(config)
    stats = await orchestrator.run_full_pipeline()

    print(f"총 생성: {stats.total_final}건")
    print(f"예상 비용: ${stats.total_cost_usd:.2f}")
    ```
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        settings: Optional[Settings] = None,
    ):
        """
        오케스트레이터 초기화

        Args:
            config: 파이프라인 설정
            settings: 기본 설정
        """
        self.config = config or PipelineConfig()
        self.settings = settings or get_settings()

        # 생성기
        self.crisis_generator = CrisisGenerator(settings=self.settings)
        self.general_generator = GeneralGenerator(settings=self.settings)

        # 검증기
        self.quality_checker = QualityChecker(
            settings=self.settings,
            criteria=EvaluationCriteria(
                min_empathy_score=3,
                min_korean_naturalness=3,
            )
        )

        # 데이터 저장소
        self.sessions: dict[str, list[CounselingSession]] = {
            "track_a": [],
            "track_b": [],
            "validated": [],
            "final": [],
        }
        self.scores: list[QualityScore] = []
        self.stats = PipelineStats()

        # 디렉토리 설정
        self.output_base = Path("data")

        logger.info("PipelineOrchestrator 초기화 완료")

    async def run_full_pipeline(self) -> PipelineStats:
        """
        전체 파이프라인 실행

        Returns:
            실행 통계
        """
        logger.info("=" * 60)
        logger.info("파이프라인 시작")
        logger.info("=" * 60)

        try:
            # 1단계: Track A (고위험군) 데이터 생성
            await self._run_track_a()

            # 2단계: Few-shot 예시 준비
            self._prepare_few_shot_examples()

            # 3단계: Track B (일반상담) 데이터 생성
            await self._run_track_b()

            # 4단계: 품질 검증
            if self.config.validate:
                await self._run_validation()

            # 5단계: 후처리
            self._run_postprocessing()

            # 6단계: 최종 데이터셋 저장
            self._save_final_dataset()

            # 완료
            self.stats.completed_at = datetime.now()
            self._print_summary()

        except Exception as e:
            logger.error(f"파이프라인 오류: {e}")
            raise

        return self.stats

    async def _run_track_a(self) -> None:
        """Track A (고위험군) 데이터 생성"""
        logger.info("\n[1/6] Track A (고위험군) 데이터 생성")
        logger.info("-" * 40)

        for category in TRACK_A_CATEGORIES:
            logger.info(f"  카테고리: {category.value}")

            sessions = await self.crisis_generator.generate_batch(
                category=category,
                count=self.config.track_a_count,
                min_turns=self.config.min_turns,
                max_turns=self.config.max_turns,
            )

            self.sessions["track_a"].extend(sessions)
            self.stats.track_a_generated += len(sessions)
            self.stats.by_category[category.value] = {
                "track": "A",
                "generated": len(sessions),
            }

            logger.info(f"    생성 완료: {len(sessions)}건")

        logger.info(f"Track A 총 생성: {self.stats.track_a_generated}건")

    def _prepare_few_shot_examples(self) -> None:
        """Track B를 위한 Few-shot 예시 준비"""
        logger.info("\n[2/6] Few-shot 예시 준비")
        logger.info("-" * 40)

        # Track A에서 생성된 고품질 세션을 Few-shot 예시로 사용
        # (실제로는 품질 검증 후 최상위 세션 선택 권장)
        for category in TRACK_B_CATEGORIES:
            # 유사한 Track A 카테고리에서 예시 선택
            # 여기서는 간단히 Track A 세션 중 일부를 사용
            examples = self.sessions["track_a"][:self.config.few_shot_examples_per_category]

            if examples:
                for example in examples:
                    # 카테고리는 다르지만 상담 패턴은 참고 가능
                    self.general_generator.add_few_shot_example(category, example)

        logger.info(f"Few-shot 예시 준비 완료")

    async def _run_track_b(self) -> None:
        """Track B (일반상담) 데이터 생성"""
        logger.info("\n[3/6] Track B (일반상담) 데이터 생성")
        logger.info("-" * 40)

        for category in TRACK_B_CATEGORIES:
            logger.info(f"  카테고리: {category.value}")

            sessions = await self.general_generator.generate_batch(
                category=category,
                count=self.config.track_b_count,
                min_turns=self.config.min_turns,
                max_turns=self.config.max_turns,
                use_few_shot=True,
            )

            self.sessions["track_b"].extend(sessions)
            self.stats.track_b_generated += len(sessions)
            self.stats.by_category[category.value] = {
                "track": "B",
                "generated": len(sessions),
            }

            logger.info(f"    생성 완료: {len(sessions)}건")

        logger.info(f"Track B 총 생성: {self.stats.track_b_generated}건")

    async def _run_validation(self) -> None:
        """품질 검증"""
        logger.info("\n[4/6] 품질 검증")
        logger.info("-" * 40)

        all_sessions = self.sessions["track_a"] + self.sessions["track_b"]

        # 샘플링 검증
        self.scores = await self.quality_checker.evaluate_batch(
            all_sessions,
            sample_rate=self.config.validation_sample_rate,
        )

        # 결과 필터링
        passed, failed = self.quality_checker.filter_by_quality(
            all_sessions,
            self.scores,
        )

        self.sessions["validated"] = passed

        # 통계 업데이트
        track_a_scores = [
            s for s in self.scores
            if any(
                sess.session_id == s.session_id
                for sess in self.sessions["track_a"]
            )
        ]
        track_b_scores = [
            s for s in self.scores
            if any(
                sess.session_id == s.session_id
                for sess in self.sessions["track_b"]
            )
        ]

        self.stats.track_a_validated = len(track_a_scores)
        self.stats.track_b_validated = len(track_b_scores)
        self.stats.track_a_passed = sum(
            1 for s in track_a_scores
            if self.quality_checker.criteria.is_acceptable(s)
        )
        self.stats.track_b_passed = sum(
            1 for s in track_b_scores
            if self.quality_checker.criteria.is_acceptable(s)
        )

        # 품질 요약 출력
        self.quality_checker.print_summary(self.scores)

    def _run_postprocessing(self) -> None:
        """후처리"""
        logger.info("\n[5/6] 후처리")
        logger.info("-" * 40)

        sessions = (
            self.sessions["validated"]
            if self.config.validate
            else self.sessions["track_a"] + self.sessions["track_b"]
        )

        # 민감정보 마스킹
        if self.config.mask_sensitive_info:
            sessions = self._mask_sensitive_info(sessions)
            logger.info("  민감정보 마스킹 완료")

        # 중복 제거
        if self.config.deduplicate:
            original_count = len(sessions)
            sessions = self._deduplicate(sessions)
            removed = original_count - len(sessions)
            logger.info(f"  중복 제거: {removed}건 제거됨")

        # 사고 과정 제거 (옵션)
        if not self.config.include_thought_trace:
            sessions = self._remove_thought_traces(sessions)
            logger.info("  사고 과정 제거 완료")

        self.sessions["final"] = sessions
        self.stats.total_final = len(sessions)

    def _mask_sensitive_info(
        self,
        sessions: list[CounselingSession]
    ) -> list[CounselingSession]:
        """민감정보 마스킹"""
        import re

        patterns = [
            (r'\d{3}-\d{4}-\d{4}', '[전화번호]'),  # 전화번호
            (r'\d{6}-\d{7}', '[주민번호]'),         # 주민번호
            (r'[가-힣]{2,4}(?=님|씨|선생)', '[이름]'),  # 이름
        ]

        for session in sessions:
            for turn in session.turns:
                for pattern, replacement in patterns:
                    turn.client.text = re.sub(pattern, replacement, turn.client.text)
                    turn.therapist.utterance = re.sub(pattern, replacement, turn.therapist.utterance)

        return sessions

    def _deduplicate(
        self,
        sessions: list[CounselingSession]
    ) -> list[CounselingSession]:
        """중복 제거 (유사도 기반)"""
        # 간단한 구현: session_id 기준
        seen = set()
        unique = []

        for session in sessions:
            if session.session_id not in seen:
                seen.add(session.session_id)
                unique.append(session)

        return unique

    def _remove_thought_traces(
        self,
        sessions: list[CounselingSession]
    ) -> list[CounselingSession]:
        """사고 과정 제거"""
        for session in sessions:
            session.metadata.pop("thought_trace", None)
            for turn in session.turns:
                turn.therapist.clinical_reasoning = ""

        return sessions

    def _save_final_dataset(self) -> None:
        """최종 데이터셋 저장"""
        logger.info("\n[6/6] 최종 데이터셋 저장")
        logger.info("-" * 40)

        # 디렉토리 생성
        final_dir = self.output_base / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 저장
        if self.config.output_format in ["json", "both"]:
            json_path = final_dir / f"korean_counseling_dataset_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metadata": {
                            "created_at": timestamp,
                            "total_sessions": len(self.sessions["final"]),
                            "config": self.config.to_dict(),
                            "stats": self.stats.to_dict(),
                        },
                        "sessions": [
                            s.model_dump(mode="json")
                            for s in self.sessions["final"]
                        ],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"  JSON 저장: {json_path}")

        # JSONL 저장
        if self.config.output_format in ["jsonl", "both"]:
            jsonl_path = final_dir / f"korean_counseling_dataset_{timestamp}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for session in self.sessions["final"]:
                    line = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
                    f.write(line + "\n")
            logger.info(f"  JSONL 저장: {jsonl_path}")

        # 품질 점수 저장
        if self.scores:
            scores_path = final_dir / f"quality_scores_{timestamp}.json"
            self.quality_checker.save_scores(self.scores, scores_path)

        # 통계 저장
        stats_path = final_dir / f"pipeline_stats_{timestamp}.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"  통계 저장: {stats_path}")

    def _print_summary(self) -> None:
        """최종 요약 출력"""
        duration = (
            self.stats.completed_at - self.stats.started_at
        ).total_seconds() if self.stats.completed_at else 0

        # 비용 계산
        self.stats.total_cost_usd = (
            self.crisis_generator.stats.get("estimated_cost", 0) +
            self.general_generator.stats.get("estimated_cost", 0)
        )

        print("\n" + "=" * 60)
        print("📊 파이프라인 실행 완료 - 최종 보고서")
        print("=" * 60)

        print(f"\n⏱️  소요 시간: {duration:.1f}초 ({duration/60:.1f}분)")

        print(f"\n📁 데이터 생성 현황")
        print(f"  Track A (고위험군): {self.stats.track_a_generated}건")
        print(f"  Track B (일반상담): {self.stats.track_b_generated}건")
        print(f"  총 생성: {self.stats.track_a_generated + self.stats.track_b_generated}건")

        if self.config.validate:
            print(f"\n✅ 품질 검증 결과")
            print(f"  Track A 통과: {self.stats.track_a_passed}/{self.stats.track_a_validated}")
            print(f"  Track B 통과: {self.stats.track_b_passed}/{self.stats.track_b_validated}")

        print(f"\n📦 최종 데이터셋: {self.stats.total_final}건")

        print(f"\n💰 예상 비용: ${self.stats.total_cost_usd:.2f} USD")

        print("\n" + "=" * 60 + "\n")

    async def run_single_track(
        self,
        track: Literal["A", "B"],
        count_per_category: Optional[int] = None,
    ) -> PipelineStats:
        """
        단일 트랙만 실행

        Args:
            track: 실행할 트랙 ("A" 또는 "B")
            count_per_category: 카테고리당 생성 수 (선택적)

        Returns:
            실행 통계
        """
        if track == "A":
            if count_per_category:
                self.config.track_a_count = count_per_category
            await self._run_track_a()
        else:
            if count_per_category:
                self.config.track_b_count = count_per_category
            await self._run_track_b()

        if self.config.validate:
            await self._run_validation()

        self._run_postprocessing()
        self._save_final_dataset()

        self.stats.completed_at = datetime.now()
        return self.stats

    async def run_single_category(
        self,
        category: CounselingCategory,
        count: int = 10,
    ) -> list[CounselingSession]:
        """
        단일 카테고리만 실행

        Args:
            category: 실행할 카테고리
            count: 생성 수

        Returns:
            생성된 세션 리스트
        """
        track = CounselingCategory.get_track(category)

        if track == "A":
            sessions = await self.crisis_generator.generate_batch(
                category=category,
                count=count,
                min_turns=self.config.min_turns,
                max_turns=self.config.max_turns,
            )
        else:
            sessions = await self.general_generator.generate_batch(
                category=category,
                count=count,
                min_turns=self.config.min_turns,
                max_turns=self.config.max_turns,
            )

        return sessions
