"""
배치 처리 모듈 (Batch Processor)

대량의 데이터 생성 작업을 효율적으로 처리합니다.
JSONL 형식의 입출력, 비동기 처리, 진행률 추적, 에러 핸들링을 지원합니다.
"""

import asyncio
import json
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

from config.settings import Settings, get_settings
from models.schemas import (
    CounselingCategory,
    CounselingSession,
    GenerationRequest,
    BatchResult,
    QualityScore,
)
from generators.crisis_generator import CrisisGenerator
from generators.general_generator import GeneralGenerator
from validators.quality_checker import QualityChecker, EvaluationCriteria

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """배치 작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """
    배치 작업 정의

    하나의 배치 작업에 대한 설정과 상태를 관리합니다.
    """
    job_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    track: Literal["A", "B", "all"] = "all"
    categories: list[CounselingCategory] = field(default_factory=list)
    count_per_category: int = 10
    min_turns: int = 5
    max_turns: int = 10
    validate: bool = True
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[BatchResult] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "track": self.track,
            "categories": [c.value for c in self.categories],
            "count_per_category": self.count_per_category,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "validate": self.validate,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class BatchProcessor:
    """
    배치 처리기

    대량의 상담 데이터를 효율적으로 생성하고 검증합니다.
    비동기 처리를 통해 처리량을 최대화하고,
    진행률 추적과 에러 핸들링을 제공합니다.

    사용 예시:
    ```python
    processor = BatchProcessor()

    # 배치 작업 생성
    job = BatchJob(
        name="일반상담 데이터셋",
        track="B",
        count_per_category=100,
        validate=True
    )

    # 실행
    result = await processor.run(job)
    print(f"생성 완료: {result.total_generated}건")
    ```
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        배치 처리기 초기화

        Args:
            settings: 설정 객체 (선택적)
            output_dir: 출력 디렉토리 (선택적)
        """
        self.settings = settings or get_settings()
        self.output_dir = output_dir or self.settings.raw_data_dir

        # 생성기
        self._crisis_generator: Optional[CrisisGenerator] = None
        self._general_generator: Optional[GeneralGenerator] = None

        # 검증기
        self._quality_checker: Optional[QualityChecker] = None

        # 작업 큐
        self.jobs: dict[str, BatchJob] = {}

        # 진행률 콜백
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None

        # Graceful shutdown 플래그
        self._shutdown_requested = False
        self._setup_signal_handlers()

        logger.info("BatchProcessor 초기화 완료")

    def _setup_signal_handlers(self) -> None:
        """시그널 핸들러 설정 (Ctrl+C 처리)"""
        def signal_handler(signum, frame):
            if not self._shutdown_requested:
                self._shutdown_requested = True
                print("\n\n⚠️  중지 요청됨 - 현재 카테고리 완료 후 안전하게 종료합니다...")
                print("   (다시 Ctrl+C를 누르면 즉시 종료)")
            else:
                print("\n❌ 강제 종료")
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, signal_handler)

    def is_shutdown_requested(self) -> bool:
        """중지 요청 여부 확인"""
        return self._shutdown_requested

    def reset_shutdown(self) -> None:
        """중지 플래그 초기화"""
        self._shutdown_requested = False

    def get_current_cost(self) -> float:
        """현재까지 사용된 예상 비용 (USD)"""
        cost = 0.0
        if self._crisis_generator:
            cost += self._crisis_generator.stats.get("estimated_cost", 0.0)
        if self._general_generator:
            cost += self._general_generator.stats.get("estimated_cost", 0.0)
        return cost

    def is_budget_exceeded(self) -> bool:
        """예산 한도 초과 여부 확인"""
        max_budget = self.settings.max_budget_usd
        if max_budget <= 0:
            return False  # 무제한
        return self.get_current_cost() >= max_budget

    def get_remaining_budget(self) -> float:
        """남은 예산 (USD, 무제한이면 -1)"""
        max_budget = self.settings.max_budget_usd
        if max_budget <= 0:
            return -1  # 무제한
        return max(0, max_budget - self.get_current_cost())

    def get_completed_categories(self, job_id: str) -> list[str]:
        """
        완료된 카테고리 목록 조회

        Args:
            job_id: 작업 ID

        Returns:
            완료된 카테고리 이름 리스트
        """
        job_dir = self.output_dir / job_id
        if not job_dir.exists():
            return []

        completed = []
        for file_path in job_dir.glob("*.jsonl"):
            # sessions.jsonl은 전체 통합 파일이므로 제외
            if file_path.stem != "sessions":
                completed.append(file_path.stem)

        return completed

    def get_resume_info(self, job_id: str) -> dict:
        """
        이어서 시작 정보 조회

        Args:
            job_id: 작업 ID

        Returns:
            이어서 시작 정보 딕셔너리
        """
        job_dir = self.output_dir / job_id
        if not job_dir.exists():
            return {"exists": False}

        completed = self.get_completed_categories(job_id)

        # 메타데이터 파일 확인
        metadata_path = job_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except:
                pass

        # 완료된 세션 수 계산
        total_sessions = 0
        for cat in completed:
            cat_file = job_dir / f"{cat}.jsonl"
            if cat_file.exists():
                with open(cat_file, "r", encoding="utf-8") as f:
                    total_sessions += sum(1 for _ in f)

        return {
            "exists": True,
            "job_id": job_id,
            "job_dir": str(job_dir),
            "completed_categories": completed,
            "total_sessions": total_sessions,
            "metadata": metadata,
        }

    @property
    def crisis_generator(self) -> CrisisGenerator:
        """위기 상담 생성기 (지연 로딩)"""
        if self._crisis_generator is None:
            self._crisis_generator = CrisisGenerator(settings=self.settings)
        return self._crisis_generator

    @property
    def general_generator(self) -> GeneralGenerator:
        """일반 상담 생성기 (지연 로딩)"""
        if self._general_generator is None:
            self._general_generator = GeneralGenerator(settings=self.settings)
        return self._general_generator

    @property
    def quality_checker(self) -> QualityChecker:
        """품질 검증기 (지연 로딩)"""
        if self._quality_checker is None:
            self._quality_checker = QualityChecker(settings=self.settings)
        return self._quality_checker

    def set_progress_callback(
        self,
        callback: Callable[[int, int, str], None]
    ) -> None:
        """
        진행률 콜백 설정

        Args:
            callback: (현재, 총계, 메시지) -> None
        """
        self.progress_callback = callback

    def _report_progress(
        self,
        current: int,
        total: int,
        message: str = ""
    ) -> None:
        """진행률 보고"""
        if self.progress_callback:
            self.progress_callback(current, total, message)
        else:
            pct = current / total * 100 if total > 0 else 0
            logger.info(f"진행률: {current}/{total} ({pct:.1f}%) {message}")

    def create_job(
        self,
        name: str,
        track: Literal["A", "B", "all"] = "all",
        categories: Optional[list[CounselingCategory]] = None,
        count_per_category: int = 10,
        min_turns: int = 5,
        max_turns: int = 10,
        validate: bool = True,
        resume_job_id: Optional[str] = None,
    ) -> BatchJob:
        """
        배치 작업 생성

        Args:
            name: 작업 이름
            track: 트랙 ("A", "B", 또는 "all")
            categories: 카테고리 리스트 (None이면 트랙 기반 자동 선택)
            count_per_category: 카테고리당 생성 수
            min_turns: 최소 턴 수
            max_turns: 최대 턴 수
            validate: 품질 검증 실행 여부
            resume_job_id: 이어서 시작할 작업 ID (선택적)

        Returns:
            생성된 BatchJob
        """
        # 카테고리 자동 선택
        if categories is None:
            if track == "A":
                from models.prompts import TRACK_A_CATEGORIES
                categories = TRACK_A_CATEGORIES
            elif track == "B":
                from models.prompts import TRACK_B_CATEGORIES
                categories = TRACK_B_CATEGORIES
            else:  # all
                from models.prompts import TRACK_A_CATEGORIES, TRACK_B_CATEGORIES
                categories = TRACK_A_CATEGORIES + TRACK_B_CATEGORIES

        # 이어서 시작인 경우 기존 job_id 사용
        if resume_job_id:
            job_id = resume_job_id
        else:
            job_id = str(uuid4())

        job = BatchJob(
            job_id=job_id,
            name=name,
            track=track,
            categories=categories,
            count_per_category=count_per_category,
            min_turns=min_turns,
            max_turns=max_turns,
            validate=validate,
        )

        self.jobs[job.job_id] = job

        if resume_job_id:
            logger.info(f"배치 작업 이어서 시작: {job.job_id} ({name})")
        else:
            logger.info(f"배치 작업 생성: {job.job_id} ({name})")

        return job

    async def run(
        self,
        job: BatchJob,
    ) -> BatchResult:
        """
        배치 작업 실행

        Args:
            job: 실행할 배치 작업

        Returns:
            BatchResult: 실행 결과
        """
        job.status = BatchStatus.RUNNING
        job.started_at = datetime.now()

        result = BatchResult(
            batch_id=job.job_id,
            total_requested=len(job.categories) * job.count_per_category,
            started_at=job.started_at,
        )

        all_sessions: list[CounselingSession] = []
        all_scores: list[QualityScore] = []

        try:
            total_categories = len(job.categories)

            # 이미 완료된 카테고리 확인 (이어서 시작 지원)
            completed_categories = self.get_completed_categories(job.job_id)
            if completed_categories:
                logger.info(f"이어서 시작: {len(completed_categories)}개 카테고리 완료됨")
                print(f"\n📋 이전 진행 상황 감지:")
                for cat_name in completed_categories:
                    print(f"   ✅ {cat_name}: 완료됨")

            for idx, category in enumerate(job.categories):
                # 이미 완료된 카테고리는 건너뛰기
                if category.value in completed_categories:
                    self._report_progress(
                        idx + 1,
                        total_categories,
                        f"카테고리 건너뛰기 (완료됨): {category.value}"
                    )
                    continue

                self._report_progress(
                    idx,
                    total_categories,
                    f"카테고리 처리 중: {category.value}"
                )

                # 트랙 결정
                track = CounselingCategory.get_track(category)

                # 생성
                if track == "A":
                    sessions = await self.crisis_generator.generate_batch(
                        category=category,
                        count=job.count_per_category,
                        min_turns=job.min_turns,
                        max_turns=job.max_turns,
                    )
                else:
                    sessions = await self.general_generator.generate_batch(
                        category=category,
                        count=job.count_per_category,
                        min_turns=job.min_turns,
                        max_turns=job.max_turns,
                    )

                all_sessions.extend(sessions)
                result.total_generated += len(sessions)
                result.categories[category.value] = len(sessions)

                # 검증
                if job.validate and sessions:
                    scores = await self.quality_checker.evaluate_batch(
                        sessions,
                        sample_rate=self.settings.validation_sample_rate,
                    )
                    all_scores.extend(scores)

                    passed = sum(
                        1 for s in scores
                        if self.quality_checker.criteria.is_acceptable(s)
                    )
                    result.total_validated += passed
                    result.total_rejected += len(scores) - passed

                # 카테고리별 즉시 저장 (데이터 손실 방지)
                if sessions:
                    self._save_category_sessions(job, category, sessions)
                    logger.info(f"카테고리 저장 완료: {category.value} ({len(sessions)}건)")

                # 중지 요청 확인
                if self._shutdown_requested:
                    print(f"\n⏹️  중지됨 - {idx + 1}/{total_categories} 카테고리 완료")
                    print(f"   저장된 데이터: {result.total_generated}건")
                    job.status = BatchStatus.COMPLETED
                    break

                # 예산 한도 확인
                if self.is_budget_exceeded():
                    current_cost = self.get_current_cost()
                    max_budget = self.settings.max_budget_usd
                    print(f"\n💰 예산 한도 도달 - ${current_cost:.2f} / ${max_budget:.2f}")
                    print(f"   완료된 카테고리: {idx + 1}/{total_categories}")
                    print(f"   저장된 데이터: {result.total_generated}건")
                    job.status = BatchStatus.COMPLETED
                    break

            # 최종 결과 저장 (메타데이터, 품질 점수)
            if all_sessions:
                self._save_results(job, all_sessions, all_scores, result)

            # 통계 계산
            if all_scores:
                result.average_quality_score = sum(
                    s.overall_score for s in all_scores
                ) / len(all_scores)

            # 비용 추정
            result.estimated_cost_usd = (
                self.crisis_generator.stats.get("estimated_cost", 0) +
                self.general_generator.stats.get("estimated_cost", 0)
            )

            # 완료
            job.status = BatchStatus.COMPLETED
            job.completed_at = datetime.now()
            result.completed_at = job.completed_at
            result.duration_seconds = (
                job.completed_at - job.started_at
            ).total_seconds()
            job.result = result

            self._report_progress(
                total_categories,
                total_categories,
                "완료!"
            )

            logger.info(
                f"배치 작업 완료: {job.job_id} "
                f"(생성: {result.total_generated}, "
                f"검증 통과: {result.total_validated})"
            )

        except Exception as e:
            job.status = BatchStatus.FAILED
            job.error = str(e)
            result.errors.append(str(e))
            logger.error(f"배치 작업 실패: {e}")

        return result

    def _save_category_sessions(
        self,
        job: BatchJob,
        category: CounselingCategory,
        sessions: list[CounselingSession],
    ) -> Path:
        """카테고리별 세션 즉시 저장 (데이터 손실 방지)"""
        output_dir = self.output_dir / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 카테고리별 파일로 저장
        category_path = output_dir / f"{category.value}.jsonl"
        with open(category_path, "w", encoding="utf-8") as f:
            for session in sessions:
                line = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
                f.write(line + "\n")

        return category_path

    def _save_results(
        self,
        job: BatchJob,
        sessions: list[CounselingSession],
        scores: list[QualityScore],
        result: BatchResult,
    ) -> None:
        """최종 결과 저장 (메타데이터, 통합 파일)"""
        output_dir = self.output_dir / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 세션 저장 (JSONL)
        sessions_path = output_dir / "sessions.jsonl"
        with open(sessions_path, "w", encoding="utf-8") as f:
            for session in sessions:
                line = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
                f.write(line + "\n")

        # 점수 저장
        if scores:
            scores_path = output_dir / "quality_scores.json"
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump(
                    [s.model_dump(mode="json") for s in scores],
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        # 메타데이터 저장
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job": job.to_dict(),
                    "result": result.model_dump(mode="json"),
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        logger.info(f"결과 저장 완료: {output_dir}")

    async def run_from_jsonl(
        self,
        input_path: Path,
        output_path: Path,
        validate: bool = True,
    ) -> BatchResult:
        """
        JSONL 파일에서 요청을 읽어 처리

        Args:
            input_path: 입력 JSONL 파일 경로
            output_path: 출력 경로
            validate: 검증 실행 여부

        Returns:
            처리 결과
        """
        requests = []

        # 입력 파일 읽기
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    req = GenerationRequest(**data)
                    requests.append(req)

        logger.info(f"JSONL 로드 완료: {len(requests)}개 요청")

        # 작업 생성 및 실행
        all_categories = list(set(req.category for req in requests))
        total_count = sum(req.count for req in requests)

        job = BatchJob(
            name=f"JSONL 배치: {input_path.name}",
            categories=all_categories,
            count_per_category=total_count // len(all_categories) if all_categories else 0,
            validate=validate,
        )

        return await self.run(job)

    def export_to_csv(
        self,
        sessions: list[CounselingSession],
        output_path: Path,
    ) -> Path:
        """
        세션을 CSV로 내보내기

        Args:
            sessions: 세션 리스트
            output_path: 출력 경로

        Returns:
            저장된 파일 경로
        """
        import csv

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)

            # 헤더
            writer.writerow([
                "session_id",
                "category",
                "risk_level",
                "turn_number",
                "client_text",
                "client_emotion",
                "therapist_reasoning",
                "therapist_utterance",
                "empathy_technique",
            ])

            # 데이터
            for session in sessions:
                for turn in session.turns:
                    writer.writerow([
                        session.session_id,
                        session.category.value,
                        session.risk_level.value,
                        turn.turn_number,
                        turn.client.text,
                        turn.client.emotional_state,
                        turn.therapist.clinical_reasoning,
                        turn.therapist.utterance,
                        turn.therapist.empathy_technique.value,
                    ])

        logger.info(f"CSV 저장 완료: {output_path}")
        return output_path

    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """작업 상태 조회"""
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[BatchJob]:
        """모든 작업 목록 반환"""
        return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """작업 취소"""
        job = self.jobs.get(job_id)
        if job and job.status in [BatchStatus.PENDING, BatchStatus.RUNNING]:
            job.status = BatchStatus.CANCELLED
            logger.info(f"작업 취소됨: {job_id}")
            return True
        return False


# ==================== JSONL 유틸리티 ====================

def load_sessions_from_jsonl(path: Path) -> list[CounselingSession]:
    """JSONL 또는 JSON 파일에서 세션 로드 (디렉토리도 지원)"""
    sessions = []
    path = Path(path)

    # 디렉토리인 경우 모든 JSON/JSONL 파일 로드
    if path.is_dir():
        files = list(path.glob("*.json")) + list(path.glob("*.jsonl"))
        for file_path in files:
            sessions.extend(_load_single_file(file_path))
        return sessions

    # 단일 파일인 경우
    return _load_single_file(path)


def _load_single_file(path: Path) -> list[CounselingSession]:
    """단일 파일에서 세션 로드"""
    sessions = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            # JSONL 형식 (여러 줄)
            if "\n" in content and not content.startswith("["):
                for line in content.split("\n"):
                    if line.strip():
                        data = json.loads(line)
                        sessions.append(CounselingSession(**data))
            # JSON 배열 형식
            elif content.startswith("["):
                data_list = json.loads(content)
                for data in data_list:
                    sessions.append(CounselingSession(**data))
            # 단일 JSON 객체
            else:
                data = json.loads(content)
                sessions.append(CounselingSession(**data))

    except Exception as e:
        logger.error(f"파일 로드 오류 ({path}): {e}")

    return sessions


def save_sessions_to_jsonl(sessions: list[CounselingSession], path: Path) -> None:
    """세션을 JSONL 파일로 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for session in sessions:
            line = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
            f.write(line + "\n")
