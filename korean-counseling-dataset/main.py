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
) -> None:
    """배치 생성 모드"""
    print("\n" + "=" * 60)
    if resume:
        print("📦 배치 생성 모드 (이어서 시작)")
    else:
        print("📦 배치 생성 모드")
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

    print(f"\n📋 설정:")
    print(f"  - 트랙: {track}")
    print(f"  - 카테고리: {[c.value for c in categories]}")
    print(f"  - 카테고리당 생성 수: {count}")
    print(f"  - 총 예상 생성: {len(categories) * count}건")
    print(f"  - 품질 검증: {'예' if validate else '아니오'}")

    # 배치 처리기 생성 (출력 경로 설정)
    output_path = Path(output) if output else Path("data/raw")
    print(f"  - 출력 경로: {output_path.absolute()}")
    processor = BatchProcessor(output_dir=output_path)

    # 이어서 시작 정보 확인
    if resume:
        resume_info = processor.get_resume_info(resume)
        if resume_info["exists"]:
            print(f"\n📋 이전 작업 발견:")
            print(f"  - 작업 ID: {resume}")
            print(f"  - 완료된 카테고리: {resume_info['completed_categories']}")
            print(f"  - 저장된 세션: {resume_info['total_sessions']}건")
        else:
            print(f"\n⚠️  작업 ID '{resume}'를 찾을 수 없습니다. 새로 시작합니다.")
            resume = None

    # 예산 한도 표시
    settings = get_settings()
    if settings.max_budget_usd > 0:
        print(f"  - 예산 한도: ${settings.max_budget_usd:.2f}")
    else:
        print(f"  - 예산 한도: 무제한")

    # 작업 생성
    job = processor.create_job(
        name=f"배치 생성 - {track}",
        track=track,
        categories=categories,
        count_per_category=count,
        min_turns=min_turns,
        max_turns=max_turns,
        validate=validate,
        resume_job_id=resume,
    )

    if resume:
        print(f"\n⏳ 배치 처리 이어서 시작... (작업 ID: {job.job_id})")
    else:
        print(f"\n⏳ 배치 처리 시작... (작업 ID: {job.job_id})")

    # 실행
    result = await processor.run(job)

    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 배치 처리 결과")
    print("=" * 60)
    print(f"  - 총 요청: {result.total_requested}건")
    print(f"  - 생성 성공: {result.total_generated}건")
    print(f"  - 생성 실패: {result.total_failed}건")
    if validate:
        print(f"  - 검증 통과: {result.total_validated}건")
        print(f"  - 검증 탈락: {result.total_rejected}건")
        print(f"  - 평균 품질: {result.average_quality_score:.2f}")
    print(f"  - 소요 시간: {result.duration_seconds:.1f}초")
    print(f"  - 예상 비용: ${result.estimated_cost_usd:.2f}")

    if result.errors:
        print(f"\n⚠️  오류 목록:")
        for err in result.errors[:5]:
            print(f"    - {err}")


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

    # 이어서 시작: 완료된 작업 확인
    completed_tasks = set()
    if job_dir.exists():
        # 분포 모드: 카테고리/턴타입.jsonl 구조
        for cat_dir in job_dir.iterdir():
            if cat_dir.is_dir():
                for file_path in cat_dir.glob("*.jsonl"):
                    completed_tasks.add(f"{cat_dir.name}/{file_path.stem}")
        # 일반 모드: 카테고리.jsonl 구조
        for file_path in job_dir.glob("*.jsonl"):
            if file_path.stem not in ["sessions", "responses"]:
                completed_tasks.add(file_path.stem)

        if completed_tasks:
            print(f"\n📋 이전 작업 발견 (작업 ID: {job_id}):")
            print(f"   완료된 작업: {len(completed_tasks)}개")

    # 작업 목록 생성
    tasks = []
    for cat in categories:
        if distribution:
            for d in dist_config:
                task_id = f"{cat.value}/{d['name']}"
                if task_id not in completed_tasks:
                    # 특정 턴 타입만 지정 시 비율 무시하고 count 사용
                    if turn_type:
                        task_count = count
                    else:
                        task_count = max(1, int(count * d['ratio']))
                    tasks.append({
                        "category": cat,
                        "turn_type": d['name'],
                        "min_turns": d['min'],
                        "max_turns": d['max'],
                        "count": task_count,
                        "task_id": task_id,
                    })
        else:
            if cat.value not in completed_tasks:
                tasks.append({
                    "category": cat,
                    "turn_type": None,
                    "min_turns": min_turns,
                    "max_turns": max_turns,
                    "count": count,
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

    # 비용 추정 (Batch 가격)
    estimated_input_tokens = total_count * 3000
    estimated_output_tokens = total_count * 6000
    estimated_cost = (
        (estimated_input_tokens / 1_000_000) * 1.00 +  # 입력 $1.00/1M
        (estimated_output_tokens / 1_000_000) * 6.00   # 출력 $6.00/1M
    )
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

        print(f"\n{'=' * 60}")
        if turn_type:
            print(f"📂 작업 [{task_idx + 1}/{len(tasks)}]: {cat.value}/{turn_type}")
            print(f"   {CounselingCategory.get_korean_name(cat)} - {task['min_turns']}~{task['max_turns']}턴")
        else:
            print(f"📂 작업 [{task_idx + 1}/{len(tasks)}]: {cat.value}")
            print(f"   ({CounselingCategory.get_korean_name(cat)})")
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
