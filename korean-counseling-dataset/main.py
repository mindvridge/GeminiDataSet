#!/usr/bin/env python3
"""
한국어 심리상담 데이터셋 구축 파이프라인

Gemini 3 API를 활용하여 고품질 심리상담 데이터셋을 생성합니다.
Track A(고위험군)와 Track B(일반상담)로 나뉘어 각각 Pro와 Flash 모델을 사용합니다.

사용 예시:
    # 단일 위기상담 데이터 생성
    python main.py --mode single --track A --category suicide_crisis

    # 일반상담 100건 배치 생성
    python main.py --mode batch --track B --count 100 --validate

    # 전체 카테고리 데이터셋 구축
    python main.py --mode batch --track all --count 1000 --validate --output ./data/final/

    # 전체 파이프라인 실행
    python main.py --mode pipeline --track-a-count 50 --track-b-count 200
"""

import argparse
import asyncio
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings, configure_environment
from config.safety_config import print_security_guidelines
from models.schemas import CounselingCategory, CounselingSession
from models.prompts import TRACK_A_CATEGORIES, TRACK_B_CATEGORIES
from generators.crisis_generator import CrisisGenerator
from generators.general_generator import GeneralGenerator
from generators.batch_client import BatchClient
from validators.quality_checker import QualityChecker, EvaluationCriteria
from pipeline.batch_processor import BatchProcessor, BatchJob
from pipeline.orchestrator import PipelineOrchestrator, PipelineConfig
from models.prompts import get_system_prompt, get_scenario_prompt


# 턴 분포 설정
TURN_DISTRIBUTIONS = {
    "balanced": [
        {"name": "short", "min": 5, "max": 10, "ratio": 0.20},   # 20%
        {"name": "medium", "min": 15, "max": 25, "ratio": 0.60}, # 60%
        {"name": "long", "min": 30, "max": 40, "ratio": 0.20},   # 20%
    ],
    "short-focus": [
        {"name": "short", "min": 5, "max": 10, "ratio": 0.60},
        {"name": "medium", "min": 15, "max": 25, "ratio": 0.30},
        {"name": "long", "min": 30, "max": 40, "ratio": 0.10},
    ],
    "long-focus": [
        {"name": "short", "min": 5, "max": 10, "ratio": 0.10},
        {"name": "medium", "min": 15, "max": 25, "ratio": 0.30},
        {"name": "long", "min": 30, "max": 40, "ratio": 0.60},
    ],
}


# 로깅 설정
def setup_logging(level: str = "INFO") -> None:
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


class ProgressIndicator:
    """
    진행률 표시기 - 스피너, 경과 시간, 현재 상태 표시
    """
    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "처리 중"):
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = None
        self.step = ""
        self.lock = threading.Lock()

    def _format_time(self, seconds: float) -> str:
        """경과 시간 포맷"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}분 {secs:.0f}초"

    def _spin(self):
        """스피너 애니메이션 실행"""
        idx = 0
        while self.running:
            with self.lock:
                elapsed = time.time() - self.start_time
                spinner = self.SPINNER_CHARS[idx % len(self.SPINNER_CHARS)]
                step_text = f" - {self.step}" if self.step else ""
                # \r로 같은 줄에 덮어쓰기
                print(f"\r{spinner} {self.message}{step_text} ({self._format_time(elapsed)})   ", end="", flush=True)
            idx += 1
            time.sleep(0.1)
        # 마지막 줄 정리
        print("\r" + " " * 80 + "\r", end="", flush=True)

    def set_step(self, step: str):
        """현재 단계 업데이트"""
        with self.lock:
            self.step = step

    def start(self):
        """진행률 표시 시작"""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        """진행률 표시 중지, 경과 시간 반환"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        elapsed = time.time() - self.start_time if self.start_time else 0
        return elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def get_category(category_name: str) -> CounselingCategory:
    """문자열에서 CounselingCategory 반환"""
    try:
        return CounselingCategory(category_name)
    except ValueError:
        available = [c.value for c in CounselingCategory]
        raise ValueError(
            f"유효하지 않은 카테고리: {category_name}\n"
            f"사용 가능한 카테고리: {available}"
        )


async def run_single_mode(
    track: str,
    category: str,
    min_turns: int = 5,
    max_turns: int = 10,
    output: Optional[str] = None,
) -> Optional[CounselingSession]:
    """단일 세션 생성 모드"""
    print("\n" + "=" * 60)
    print("🔬 단일 세션 생성 모드")
    print("=" * 60)

    cat = get_category(category)
    actual_track = CounselingCategory.get_track(cat)

    print(f"\n📋 설정:")
    print(f"  - 카테고리: {cat.value} ({CounselingCategory.get_korean_name(cat)})")
    print(f"  - 트랙: {actual_track}")
    print(f"  - 턴 수: {min_turns}~{max_turns}")

    # 생성기 선택
    if actual_track == "A":
        generator = CrisisGenerator()
        print("\n⚠️  고위험군 데이터 생성 - BLOCK_NONE 설정 적용")
    else:
        generator = GeneralGenerator()
        print("\n✅ 일반상담 데이터 생성")

    # 진행률 표시기 시작
    progress = ProgressIndicator("세션 생성 중")
    progress.start()

    try:
        # 프롬프트 구성 단계
        progress.set_step("프롬프트 구성")
        await asyncio.sleep(0.1)  # UI 업데이트 여유

        # API 호출 단계
        progress.set_step("Gemini API 호출")

        session = await generator.generate_session(
            category=cat,
            min_turns=min_turns,
            max_turns=max_turns,
        )

        # 응답 처리 단계
        if session:
            progress.set_step("응답 파싱")
            await asyncio.sleep(0.1)

    finally:
        elapsed = progress.stop()

    if session:
        print(f"\n✅ 생성 완료! (소요 시간: {elapsed:.1f}초)")
        print(f"  - 세션 ID: {session.session_id}")
        print(f"  - 위험 수준: {session.risk_level.value}")
        print(f"  - 대화 턴 수: {len(session.turns)}")

        # 대화 내용 미리보기
        print("\n📝 대화 미리보기 (첫 2턴):")
        print("-" * 40)
        for turn in session.turns[:2]:
            print(f"내담자: {turn.client.text[:100]}...")
            print(f"상담사: {turn.therapist.utterance[:100]}...")
            print()

        # 저장 (기본값: data/raw/)
        output_path = Path(output) if output else Path("data/raw")
        saved_path = generator.save_session(session, output_path)
        print(f"\n💾 저장 완료: {saved_path}")

        return session
    else:
        print(f"\n❌ 세션 생성 실패 (소요 시간: {elapsed:.1f}초)")
        return None


async def run_batch_mode(
    track: str,
    category: Optional[str] = None,
    count: int = 10,
    min_turns: int = 5,
    max_turns: int = 10,
    validate: bool = True,
    output: Optional[str] = None,
    resume: Optional[str] = None,
    distribution: Optional[str] = None,
    turn_type: Optional[str] = None,
) -> None:
    """
    배치 생성 모드 (Standard API)

    BLOCK_NONE 안전 설정이 적용되며, --distribution 옵션으로 턴 분포 적용 가능.
    저장 구조: <output>/<job_id>/<category>/<turn_type>.jsonl
    """
    import json
    import signal
    from uuid import uuid4

    print("\n" + "=" * 60)
    if resume:
        print("📦 배치 생성 모드 (이어서 시작) - BLOCK_NONE 적용")
    else:
        print("📦 배치 생성 모드 - BLOCK_NONE 적용")
    print("=" * 60)

    # 카테고리 결정
    if category:
        categories = [get_category(category)]
    elif track == "A":
        categories = TRACK_A_CATEGORIES
    elif track == "B":
        categories = TRACK_B_CATEGORIES
    else:  # all
        categories = TRACK_A_CATEGORIES + TRACK_B_CATEGORIES

    # 출력 경로 설정
    output_path = Path(output) if output else Path("data/raw")

    # 작업 ID 결정 (resume 또는 새로 생성)
    if resume:
        job_id = resume
    else:
        job_id = str(uuid4())

    job_dir = output_path / job_id

    # 분포 설정
    if distribution:
        dist_config = TURN_DISTRIBUTIONS[distribution]
        if turn_type:
            dist_config = [d for d in dist_config if d['name'] == turn_type]
            print(f"\n📊 턴 분포: {distribution} (필터: {turn_type})")
        else:
            print(f"\n📊 턴 분포: {distribution}")
        for d in dist_config:
            print(f"   - {d['name']}: {d['min']}~{d['max']}턴 ({int(d['ratio']*100)}%)")
    else:
        dist_config = None

    # 이어서 시작: 완료된 작업 및 기존 세션 수 확인
    existing_counts = {}  # task_id -> existing session count
    if job_dir.exists():
        # 분포 모드: 카테고리/턴타입.jsonl 구조
        for cat_dir in job_dir.iterdir():
            if cat_dir.is_dir():
                for file_path in cat_dir.glob("*.jsonl"):
                    task_id = f"{cat_dir.name}/{file_path.stem}"
                    # 파일 내 세션 수 계산
                    line_count = sum(1 for _ in open(file_path, encoding='utf-8'))
                    existing_counts[task_id] = line_count
        # 일반 모드: 카테고리.jsonl 구조
        for file_path in job_dir.glob("*.jsonl"):
            if file_path.stem not in ["sessions", "responses"]:
                line_count = sum(1 for _ in open(file_path, encoding='utf-8'))
                existing_counts[file_path.stem] = line_count

        if existing_counts:
            print(f"\n📋 이전 작업 발견 (작업 ID: {job_id}):")
            for task_id, cnt in existing_counts.items():
                print(f"   - {task_id}: {cnt}건")

    # 작업 목록 생성 (부족한 수량만 생성)
    tasks = []
    for cat in categories:
        if distribution:
            for d in dist_config:
                task_id = f"{cat.value}/{d['name']}"
                if turn_type:
                    target_count = count
                else:
                    target_count = max(1, int(count * d['ratio']))

                existing = existing_counts.get(task_id, 0)
                remaining = target_count - existing

                if remaining > 0:
                    tasks.append({
                        "category": cat,
                        "turn_type": d['name'],
                        "min_turns": d['min'],
                        "max_turns": d['max'],
                        "count": remaining,
                        "target_count": target_count,
                        "existing_count": existing,
                        "task_id": task_id,
                    })
        else:
            existing = existing_counts.get(cat.value, 0)
            remaining = count - existing

            if remaining > 0:
                tasks.append({
                    "category": cat,
                    "turn_type": None,
                    "min_turns": min_turns,
                    "max_turns": max_turns,
                    "count": remaining,
                    "target_count": count,
                    "existing_count": existing,
                    "task_id": cat.value,
                })

    if not tasks:
        print(f"\n✅ 모든 작업이 이미 완료되었습니다!")
        return

    total_count = sum(t['count'] for t in tasks)

    print(f"\n📋 설정:")
    print(f"  - 트랙: {track}")
    print(f"  - 작업 ID: {job_id}")
    print(f"  - 카테고리: {[c.value for c in categories]}")
    print(f"  - 남은 작업: {len(tasks)}개")
    print(f"  - 총 예상 생성: {total_count}건")
    print(f"  - 품질 검증: {'예' if validate else '아니오'}")

    # 비용 추정 (Standard 가격) - 턴 수 기반 계산
    TOKENS_INPUT_BASE = 700
    TOKENS_PER_TURN = 300

    estimated_input_tokens = 0
    estimated_output_tokens = 0

    for task in tasks:
        avg_turns = (task['min_turns'] + task['max_turns']) / 2
        task_input = task['count'] * TOKENS_INPUT_BASE
        task_output = task['count'] * avg_turns * TOKENS_PER_TURN
        estimated_input_tokens += task_input
        estimated_output_tokens += task_output

    # Standard API 가격 (Batch의 2배)
    estimated_cost = (
        (estimated_input_tokens / 1_000_000) * 2.00 +  # 입력 $2.00/1M
        (estimated_output_tokens / 1_000_000) * 12.00  # 출력 $12.00/1M
    )

    print(f"  - 예상 입력 토큰: {estimated_input_tokens:,}")
    print(f"  - 예상 출력 토큰: {estimated_output_tokens:,}")
    print(f"  - 예상 비용: ${estimated_cost:.2f} (Standard 가격)")

    # 예산 한도 표시
    settings = get_settings()
    if settings.max_budget_usd > 0:
        print(f"  - 예산 한도: ${settings.max_budget_usd:.2f}")

    print(f"\n⚠️  Track A 고위험군은 BLOCK_NONE 안전 설정이 적용됩니다.")
    print(f"✅ 각 작업 완료 시 즉시 저장됩니다.")

    # 확인
    confirm = input("\n계속 진행하시겠습니까? (y/N): ")
    if confirm.lower() != 'y':
        print("취소되었습니다.")
        return

    # 출력 경로 생성
    job_dir.mkdir(parents=True, exist_ok=True)

    # Graceful shutdown 설정
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n\n⚠️  강제 종료...")
            sys.exit(1)
        shutdown_requested = True
        print("\n\n⚠️  현재 작업 완료 후 종료합니다. (다시 Ctrl+C: 강제 종료)")

    signal.signal(signal.SIGINT, signal_handler)

    # 생성기 초기화
    crisis_gen = CrisisGenerator(settings=settings)
    general_gen = GeneralGenerator(settings=settings)
    quality_checker = QualityChecker() if validate else None

    # 결과 집계
    total_generated = 0
    total_failed = 0
    total_validated = 0
    total_rejected = 0
    total_cost_result = 0.0
    last_task_idx = 0

    # 작업별 처리
    for task_idx, task in enumerate(tasks):
        last_task_idx = task_idx
        if shutdown_requested:
            print(f"\n⚠️  사용자 요청으로 중지됨")
            break

        cat = task['category']
        t_type = task['turn_type']
        task_count = task['count']
        existing_count = task.get('existing_count', 0)
        target_count = task.get('target_count', task_count)

        print(f"\n{'=' * 60}")
        if t_type:
            print(f"📂 작업 [{task_idx + 1}/{len(tasks)}]: {cat.value}/{t_type}")
            if existing_count > 0:
                print(f"   {CounselingCategory.get_korean_name(cat)} - {task['min_turns']}~{task['max_turns']}턴")
                print(f"   기존: {existing_count}건 → 추가 생성: {task_count}건 (목표: {target_count}건)")
            else:
                print(f"   {CounselingCategory.get_korean_name(cat)} - {task['min_turns']}~{task['max_turns']}턴, {task_count}건")
        else:
            print(f"📂 작업 [{task_idx + 1}/{len(tasks)}]: {cat.value}")
            if existing_count > 0:
                print(f"   기존: {existing_count}건 → 추가 생성: {task_count}건 (목표: {target_count}건)")
            else:
                print(f"   ({CounselingCategory.get_korean_name(cat)}) - {task_count}건")
        print("=" * 60)

        # 생성기 선택
        track_for_cat = CounselingCategory.get_track(cat)
        if track_for_cat == "A":
            generator = crisis_gen
        else:
            generator = general_gen

        # 세션 생성
        print(f"⏳ 세션 생성 중... ({task_count}건)")
        sessions = await generator.generate_batch(
            category=cat,
            count=task_count,
            min_turns=task['min_turns'],
            max_turns=task['max_turns'],
        )

        # 품질 검증
        validated_sessions = sessions
        rejected_count = 0
        if validate and quality_checker and sessions:
            print(f"⏳ 품질 검증 중...")
            scores = await quality_checker.evaluate_batch(sessions, sample_rate=1.0)
            validated_sessions = []
            for session, score in zip(sessions, scores):
                if score.overall_score >= 0.6:  # 60% 이상 통과
                    validated_sessions.append(session)
                else:
                    rejected_count += 1

        # 결과 저장 (분포 모드: 카테고리/턴타입.jsonl, 일반 모드: 카테고리.jsonl)
        if t_type:
            cat_dir = job_dir / cat.value
            cat_dir.mkdir(parents=True, exist_ok=True)
            save_file = cat_dir / f"{t_type}.jsonl"
        else:
            save_file = job_dir / f"{cat.value}.jsonl"

        with open(save_file, "w", encoding="utf-8") as f:
            for session in validated_sessions:
                line = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
                f.write(line + "\n")

        # 통계
        gen_cost = generator.stats.get("estimated_cost", 0.0)
        success_count = len(validated_sessions)
        fail_count = task_count - len(sessions)

        if t_type:
            print(f"\n✅ 작업 완료: {cat.value}/{t_type}")
        else:
            print(f"\n✅ 작업 완료: {cat.value}")
        print(f"   - 생성: {len(sessions)}건")
        print(f"   - 검증 통과: {success_count}건")
        if validate:
            print(f"   - 검증 탈락: {rejected_count}건")
        print(f"   - 저장: {save_file}")

        total_generated += len(sessions)
        total_failed += fail_count
        total_validated += success_count
        total_rejected += rejected_count

    # 최종 결과
    print(f"\n{'=' * 60}")
    print("📊 배치 처리 완료")
    print("=" * 60)
    print(f"  - 작업 ID: {job_id}")
    print(f"  - 완료 작업: {len(completed_tasks) + last_task_idx + 1 - (1 if shutdown_requested else 0)}개")
    print(f"  - 총 생성: {total_generated}건")
    print(f"  - 총 실패: {total_failed}건")
    if validate:
        print(f"  - 검증 통과: {total_validated}건")
        print(f"  - 검증 탈락: {total_rejected}건")
    print(f"  - 저장 위치: {job_dir}")

    # 남은 작업 안내
    remaining_tasks = tasks[last_task_idx + (0 if shutdown_requested else 1):]
    if remaining_tasks:
        print(f"\n⚠️  남은 작업: {len(remaining_tasks)}개")
        print(f"이어서 시작하려면:")
        dist_opt = f" --distribution {distribution}" if distribution else ""
        turn_opt = f" --turn-type {turn_type}" if turn_type else ""
        cat_opt = f" --category {category}" if category else ""
        print(f"  python main.py --mode batch --track {track} --count {count}{dist_opt}{turn_opt}{cat_opt} --output {output_path} --resume {job_id}")


async def run_pipeline_mode(
    track_a_count: int = 50,
    track_b_count: int = 200,
    min_turns: int = 5,
    max_turns: int = 10,
    validate: bool = True,
    output: Optional[str] = None,
) -> None:
    """전체 파이프라인 실행 모드"""
    print("\n" + "=" * 60)
    print("🚀 전체 파이프라인 실행 모드")
    print("=" * 60)

    # 보안 가이드라인 출력
    print_security_guidelines()

    config = PipelineConfig(
        track_a_count=track_a_count,
        track_b_count=track_b_count,
        min_turns=min_turns,
        max_turns=max_turns,
        validate=validate,
        output_format="both",
    )

    print(f"\n📋 파이프라인 설정:")
    print(f"  - Track A (고위험군) 카테고리당: {track_a_count}건")
    print(f"  - Track B (일반상담) 카테고리당: {track_b_count}건")
    print(f"  - 총 예상 생성: {len(TRACK_A_CATEGORIES) * track_a_count + len(TRACK_B_CATEGORIES) * track_b_count}건")
    print(f"  - 품질 검증: {'예' if validate else '아니오'}")

    # 확인
    confirm = input("\n⚠️  파이프라인을 시작하시겠습니까? (y/N): ")
    if confirm.lower() != 'y':
        print("취소되었습니다.")
        return

    orchestrator = PipelineOrchestrator(config=config)

    print("\n⏳ 파이프라인 실행 중...")
    stats = await orchestrator.run_full_pipeline()

    print("\n✅ 파이프라인 완료!")


async def run_validate_mode(
    input_path: str,
    sample_rate: float = 0.2,
    output: Optional[str] = None,
) -> None:
    """검증 전용 모드"""
    print("\n" + "=" * 60)
    print("🔍 품질 검증 모드")
    print("=" * 60)

    from pipeline.batch_processor import load_sessions_from_jsonl

    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        return

    print(f"\n📋 설정:")
    print(f"  - 입력 파일: {input_path}")
    print(f"  - 샘플링 비율: {sample_rate * 100:.0f}%")

    # 세션 로드
    with ProgressIndicator("세션 로드 중") as progress:
        sessions = load_sessions_from_jsonl(input_path)
    print(f"✅ 로드 완료: {len(sessions)}건")

    if not sessions:
        print("\n⚠️  검증할 세션이 없습니다.")
        print("   먼저 데이터를 생성하세요:")
        print("   python main.py --mode single --track B --category career")
        return

    # 검증
    checker = QualityChecker()

    evaluated_count = max(1, int(len(sessions) * sample_rate))
    print(f"\n📊 품질 검증 시작: {evaluated_count}건 평가 예정")

    progress = ProgressIndicator("품질 검증 중")
    progress.start()
    try:
        scores = await checker.evaluate_batch(sessions, sample_rate=sample_rate)
    finally:
        elapsed = progress.stop()
    print(f"✅ 검증 완료 (소요 시간: {elapsed:.1f}초)")

    # 결과 출력
    checker.print_summary(scores)

    # 저장
    if output:
        output_path = Path(output)
        checker.save_scores(scores, output_path)
        print(f"\n💾 검증 결과 저장: {output_path}")


async def run_batch_api_mode(
    track: str,
    category: Optional[str] = None,
    count: int = 1000,
    min_turns: int = 5,
    max_turns: int = 10,
    output: Optional[str] = None,
    poll_interval: int = 60,
    resume: Optional[str] = None,
    distribution: Optional[str] = None,
    turn_type: Optional[str] = None,
) -> None:
    """
    Batch API 모드 - 50% 비용 절감, 카테고리별 분리 실행

    각 카테고리를 개별 Batch 작업으로 제출하여 완료 시 즉시 저장.
    중단 시에도 완료된 카테고리는 보존됨.
    --distribution 옵션으로 턴 분포 적용 가능.
    """
    import json
    import signal
    from uuid import uuid4

    print("\n" + "=" * 60)
    if resume:
        print("📦 Batch API 모드 (50% 비용 절감) - 이어서 시작")
    else:
        print("📦 Batch API 모드 (50% 비용 절감)")
    print("=" * 60)

    # 카테고리 결정
    if category:
        categories = [get_category(category)]
    elif track == "A":
        categories = TRACK_A_CATEGORIES
    elif track == "B":
        categories = TRACK_B_CATEGORIES
    else:
        categories = TRACK_A_CATEGORIES + TRACK_B_CATEGORIES

    # 출력 경로 설정
    output_path = Path(output) if output else Path("data/batch")

    # 작업 ID 결정 (resume 또는 새로 생성)
    if resume:
        job_id = resume
    else:
        job_id = str(uuid4())

    job_dir = output_path / job_id

    # 분포 설정
    if distribution:
        dist_config = TURN_DISTRIBUTIONS[distribution]
        # 특정 턴 타입만 필터링
        if turn_type:
            dist_config = [d for d in dist_config if d['name'] == turn_type]
            print(f"\n📊 턴 분포: {distribution} (필터: {turn_type})")
        else:
            print(f"\n📊 턴 분포: {distribution}")
        for d in dist_config:
            print(f"   - {d['name']}: {d['min']}~{d['max']}턴 ({int(d['ratio']*100)}%)")
    else:
        dist_config = None

    # 이어서 시작: 완료된 작업 및 기존 세션 수 확인
    existing_counts = {}  # task_id -> existing session count
    if job_dir.exists():
        # 분포 모드: 카테고리/턴타입.jsonl 구조
        for cat_dir in job_dir.iterdir():
            if cat_dir.is_dir():
                for file_path in cat_dir.glob("*.jsonl"):
                    task_id = f"{cat_dir.name}/{file_path.stem}"
                    line_count = sum(1 for _ in open(file_path, encoding='utf-8'))
                    existing_counts[task_id] = line_count
        # 일반 모드: 카테고리.jsonl 구조
        for file_path in job_dir.glob("*.jsonl"):
            if file_path.stem not in ["sessions", "responses"]:
                line_count = sum(1 for _ in open(file_path, encoding='utf-8'))
                existing_counts[file_path.stem] = line_count

        if existing_counts:
            print(f"\n📋 이전 작업 발견 (작업 ID: {job_id}):")
            for task_id, cnt in existing_counts.items():
                print(f"   - {task_id}: {cnt}건")

    # 작업 목록 생성 (부족한 수량만 생성)
    tasks = []
    for cat in categories:
        if distribution:
            for d in dist_config:
                task_id = f"{cat.value}/{d['name']}"
                if turn_type:
                    target_count = count
                else:
                    target_count = max(1, int(count * d['ratio']))

                existing = existing_counts.get(task_id, 0)
                remaining = target_count - existing

                if remaining > 0:
                    tasks.append({
                        "category": cat,
                        "turn_type": d['name'],
                        "min_turns": d['min'],
                        "max_turns": d['max'],
                        "count": remaining,
                        "target_count": target_count,
                        "existing_count": existing,
                        "task_id": task_id,
                    })
        else:
            existing = existing_counts.get(cat.value, 0)
            remaining = count - existing

            if remaining > 0:
                tasks.append({
                    "category": cat,
                    "turn_type": None,
                    "min_turns": min_turns,
                    "max_turns": max_turns,
                    "count": remaining,
                    "target_count": count,
                    "existing_count": existing,
                    "task_id": cat.value,
                })

    if not tasks:
        print(f"\n✅ 모든 작업이 이미 완료되었습니다!")
        return

    total_count = sum(t['count'] for t in tasks)

    print(f"\n📋 설정:")
    print(f"  - 트랙: {track}")
    print(f"  - 작업 ID: {job_id}")
    print(f"  - 카테고리: {[c.value for c in categories]}")
    print(f"  - 남은 작업: {len(tasks)}개")
    print(f"  - 총 예상 생성: {total_count}건")

    # 비용 추정 (Batch 가격) - 턴 수 기반 계산
    # 토큰 추정 상수
    TOKENS_INPUT_BASE = 700      # 시스템 프롬프트 + 사용자 프롬프트 (고정)
    TOKENS_PER_TURN = 300        # 턴당 출력 토큰 (내담자 + 상담사 발화 + 사고과정)

    estimated_input_tokens = 0
    estimated_output_tokens = 0

    for task in tasks:
        avg_turns = (task['min_turns'] + task['max_turns']) / 2
        task_input = task['count'] * TOKENS_INPUT_BASE
        task_output = task['count'] * avg_turns * TOKENS_PER_TURN
        estimated_input_tokens += task_input
        estimated_output_tokens += task_output

    estimated_cost = (
        (estimated_input_tokens / 1_000_000) * 1.00 +  # 입력 $1.00/1M
        (estimated_output_tokens / 1_000_000) * 6.00   # 출력 $6.00/1M
    )

    print(f"  - 예상 입력 토큰: {estimated_input_tokens:,}")
    print(f"  - 예상 출력 토큰: {estimated_output_tokens:,}")
    print(f"  - 예상 비용: ${estimated_cost:.2f} (Batch 가격)")
    print(f"  - 폴링 간격: {poll_interval}초")
    print(f"\n⚠️  Batch API는 작업당 최대 24시간 소요될 수 있습니다.")
    print(f"✅ 각 작업 완료 시 즉시 저장됩니다.")

    # 확인
    confirm = input("\n계속 진행하시겠습니까? (y/N): ")
    if confirm.lower() != 'y':
        print("취소되었습니다.")
        return

    # 출력 경로 생성
    job_dir.mkdir(parents=True, exist_ok=True)

    # Graceful shutdown 설정
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n\n⚠️  강제 종료...")
            sys.exit(1)
        shutdown_requested = True
        print("\n\n⚠️  현재 작업 완료 후 종료합니다. (다시 Ctrl+C: 강제 종료)")

    signal.signal(signal.SIGINT, signal_handler)

    # 결과 추적
    total_success = 0
    total_failed = 0
    total_cost_result = 0.0
    last_task_idx = 0

    # 작업별 처리
    for task_idx, task in enumerate(tasks):
        last_task_idx = task_idx
        if shutdown_requested:
            print(f"\n⚠️  사용자 요청으로 중지됨")
            break

        cat = task['category']
        turn_type = task['turn_type']
        task_count = task['count']
        existing_count = task.get('existing_count', 0)
        target_count = task.get('target_count', task_count)

        print(f"\n{'=' * 60}")
        if turn_type:
            print(f"📂 작업 [{task_idx + 1}/{len(tasks)}]: {cat.value}/{turn_type}")
            if existing_count > 0:
                print(f"   {CounselingCategory.get_korean_name(cat)} - {task['min_turns']}~{task['max_turns']}턴")
                print(f"   기존: {existing_count}건 → 추가 생성: {task_count}건 (목표: {target_count}건)")
            else:
                print(f"   {CounselingCategory.get_korean_name(cat)} - {task['min_turns']}~{task['max_turns']}턴, {task_count}건")
        else:
            print(f"📂 작업 [{task_idx + 1}/{len(tasks)}]: {cat.value}")
            if existing_count > 0:
                print(f"   기존: {existing_count}건 → 추가 생성: {task_count}건 (목표: {target_count}건)")
            else:
                print(f"   ({CounselingCategory.get_korean_name(cat)}) - {task_count}건")
        print("=" * 60)

        # BatchClient 초기화 (카테고리별 트랙 확인)
        track_for_cat = CounselingCategory.get_track(cat)
        client = BatchClient(track=track_for_cat)

        # 프롬프트 생성
        print(f"\n⏳ 프롬프트 생성 중... ({task_count}건)")
        system_prompt = get_system_prompt(track=track_for_cat, include_thinking=True)
        prompts = []

        for i in range(task_count):
            scenario_prompt = get_scenario_prompt(
                category=cat,
                min_turns=task['min_turns'],
                max_turns=task['max_turns'],
            )
            user_prompt = f"""
다음 시나리오에 맞는 심리상담 세션을 생성해주세요.

{scenario_prompt}

반드시 지정된 JSON 형식으로 출력하세요.
자연스러운 한국어 구어체를 사용하고, 공감적 화법을 유지하세요.
"""
            prompts.append({
                "system": system_prompt,
                "user": user_prompt,
                "category": cat.value,
                "turn_type": turn_type,
            })

        # 요청 빌드
        requests = client.build_requests(prompts)

        # 배치 작업 제출
        print(f"⏳ 배치 작업 제출 중...")
        try:
            display_name = f"counseling-{cat.value}-{turn_type}-{task_count}" if turn_type else f"counseling-{cat.value}-{task_count}"
            batch_job = await client.create_batch_job(
                requests=requests,
                display_name=display_name,
            )
            print(f"✅ 제출 완료 (작업: {batch_job.job_name})")

            # 완료 대기
            print(f"⏳ 완료 대기 중... (Ctrl+C: 현재 작업 완료 후 중지)")

            def progress_callback(state: str):
                print(f"   상태: {state}")

            result = await client.wait_for_completion(
                batch_job,
                poll_interval=poll_interval,
                progress_callback=progress_callback,
            )

            # 결과 저장 (분포 모드: 카테고리/턴타입.jsonl, 일반 모드: 카테고리.jsonl)
            if turn_type:
                cat_dir = job_dir / cat.value
                cat_dir.mkdir(parents=True, exist_ok=True)
                save_file = cat_dir / f"{turn_type}.jsonl"
            else:
                save_file = job_dir / f"{cat.value}.jsonl"

            with open(save_file, "w", encoding="utf-8") as f:
                for i, resp in enumerate(result.responses):
                    line = json.dumps({
                        "index": i,
                        "category": cat.value,
                        "turn_type": turn_type,
                        "min_turns": task['min_turns'],
                        "max_turns": task['max_turns'],
                        "text": resp.get("text", ""),
                    }, ensure_ascii=False)
                    f.write(line + "\n")

            if turn_type:
                print(f"\n✅ 작업 완료: {cat.value}/{turn_type}")
            else:
                print(f"\n✅ 작업 완료: {cat.value}")
            print(f"   - 성공: {len(result.responses)}건")
            print(f"   - 실패: {len(result.errors)}건")
            print(f"   - 비용: ${result.estimated_cost:.2f}")
            print(f"   - 저장: {save_file}")

            total_success += len(result.responses)
            total_failed += len(result.errors)
            total_cost_result += result.estimated_cost

        except Exception as e:
            if turn_type:
                print(f"\n❌ 작업 실패: {cat.value}/{turn_type}")
            else:
                print(f"\n❌ 작업 실패: {cat.value}")
            print(f"   오류: {e}")
            continue

    # 최종 결과
    print(f"\n{'=' * 60}")
    print("📊 Batch API 처리 완료")
    print("=" * 60)
    print(f"  - 작업 ID: {job_id}")
    print(f"  - 완료 작업: {len(completed_tasks) + last_task_idx + 1 - (1 if shutdown_requested else 0)}개")
    print(f"  - 총 성공: {total_success}건")
    print(f"  - 총 실패: {total_failed}건")
    print(f"  - 총 비용: ${total_cost_result:.2f}")
    print(f"  - 저장 위치: {job_dir}")

    # 남은 작업 안내
    remaining_tasks = tasks[last_task_idx + (0 if shutdown_requested else 1):]
    if remaining_tasks:
        print(f"\n⚠️  남은 작업: {len(remaining_tasks)}개")
        print(f"이어서 시작하려면:")
        dist_opt = f" --distribution {distribution}" if distribution else ""
        turn_opt = f" --turn-type {turn_type}" if turn_type else ""
        cat_opt = f" --category {category}" if category else ""
        print(f"  python main.py --mode batch-api --track {track} --count {count}{dist_opt}{turn_opt}{cat_opt} --output {output_path} --resume {job_id}")


def print_categories() -> None:
    """사용 가능한 카테고리 출력"""
    print("\n📋 사용 가능한 카테고리:")
    print("\n  Track A (고위험군 - Gemini Pro):")
    for cat in TRACK_A_CATEGORIES:
        print(f"    - {cat.value}: {CounselingCategory.get_korean_name(cat)}")

    print("\n  Track B (일반상담 - Gemini Flash):")
    for cat in TRACK_B_CATEGORIES:
        print(f"    - {cat.value}: {CounselingCategory.get_korean_name(cat)}")


def main():
    """메인 엔트리포인트"""
    parser = argparse.ArgumentParser(
        description="한국어 심리상담 데이터셋 구축 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 위기상담 데이터 생성
  python main.py --mode single --track A --category suicide_crisis

  # 일반상담 100건 배치 생성 (Standard API)
  python main.py --mode batch --track B --count 100 --validate

  # Batch API로 대량 생성 (50% 비용 절감)
  python main.py --mode batch-api --track A --count 1000

  # 전체 파이프라인 실행
  python main.py --mode pipeline --track-a-count 50 --track-b-count 200

  # 품질 검증만 실행
  python main.py --mode validate --input data/raw/sessions.jsonl
        """,
    )

    # 모드 선택
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch", "batch-api", "pipeline", "validate", "list-categories"],
        default="single",
        help="실행 모드: single, batch, batch-api(50%%할인), pipeline, validate (default: single)",
    )

    # 트랙 선택
    parser.add_argument(
        "--track",
        type=str,
        choices=["A", "B", "all"],
        default="B",
        help="트랙 선택: A(고위험군), B(일반상담), all (default: B)",
    )

    # 카테고리
    parser.add_argument(
        "--category",
        type=str,
        help="특정 카테고리 지정 (예: suicide_crisis, career)",
    )

    # 생성 수
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="생성할 데이터 수 (default: 10)",
    )

    # 파이프라인 모드 전용
    parser.add_argument(
        "--track-a-count",
        type=int,
        default=50,
        help="Track A 카테고리당 생성 수 (default: 50)",
    )

    parser.add_argument(
        "--track-b-count",
        type=int,
        default=200,
        help="Track B 카테고리당 생성 수 (default: 200)",
    )

    # 턴 수
    parser.add_argument(
        "--min-turns",
        type=int,
        default=5,
        help="최소 대화 턴 수 (default: 5)",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="최대 대화 턴 수 (default: 10)",
    )

    # 턴 분포
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["balanced", "short-focus", "long-focus"],
        help="턴 분포: balanced(20/60/20), short-focus(60/30/10), long-focus(10/30/60)",
    )

    # 특정 턴 타입만 실행
    parser.add_argument(
        "--turn-type",
        type=str,
        choices=["short", "medium", "long"],
        help="특정 턴 타입만 실행 (--distribution과 함께 사용)",
    )

    # 검증
    parser.add_argument(
        "--validate",
        action="store_true",
        help="품질 검증 실행",
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.2,
        help="검증 샘플링 비율 (default: 0.2)",
    )

    # 입출력
    parser.add_argument(
        "--input",
        type=str,
        help="입력 파일 경로 (validate 모드용)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="출력 경로",
    )

    # 이어서 시작
    parser.add_argument(
        "--resume",
        type=str,
        help="이어서 시작할 작업 ID (예: e23c5ce2-ae92-490f-8add-97a370c7be5f)",
    )

    # 예산 한도
    parser.add_argument(
        "--max-budget",
        type=float,
        default=0.0,
        help="최대 예산 한도 USD (default: 0=무제한)",
    )

    # 로깅
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로깅 레벨 (default: INFO)",
    )

    args = parser.parse_args()

    # 로깅 설정
    setup_logging(args.log_level)

    # 환경 변수 설정
    configure_environment()

    # 예산 한도 설정 (명령줄 옵션 우선)
    if args.max_budget > 0:
        settings = get_settings()
        settings.max_budget_usd = args.max_budget

    # 모드별 실행
    if args.mode == "list-categories":
        print_categories()
        return

    if args.mode == "single":
        if not args.category:
            parser.error("--category 옵션이 필요합니다")

        asyncio.run(run_single_mode(
            track=args.track,
            category=args.category,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            output=args.output,
        ))

    elif args.mode == "batch":
        asyncio.run(run_batch_mode(
            track=args.track,
            category=args.category,
            count=args.count,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            validate=args.validate,
            output=args.output,
            resume=args.resume,
            distribution=args.distribution,
            turn_type=args.turn_type,
        ))

    elif args.mode == "batch-api":
        asyncio.run(run_batch_api_mode(
            track=args.track,
            category=args.category,
            count=args.count,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            output=args.output,
            resume=args.resume,
            distribution=args.distribution,
            turn_type=args.turn_type,
        ))

    elif args.mode == "pipeline":
        asyncio.run(run_pipeline_mode(
            track_a_count=args.track_a_count,
            track_b_count=args.track_b_count,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            validate=args.validate,
            output=args.output,
        ))

    elif args.mode == "validate":
        if not args.input:
            parser.error("--input 옵션이 필요합니다")

        asyncio.run(run_validate_mode(
            input_path=args.input,
            sample_rate=args.sample_rate,
            output=args.output,
        ))


if __name__ == "__main__":
    main()
