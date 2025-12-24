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
) -> None:
    """
    Batch API 모드 - 50% 비용 절감

    Standard API 대신 Batch API를 사용하여 대량 데이터 생성
    """
    print("\n" + "=" * 60)
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

    total_count = len(categories) * count

    print(f"\n📋 설정:")
    print(f"  - 트랙: {track}")
    print(f"  - 카테고리: {[c.value for c in categories]}")
    print(f"  - 카테고리당 생성 수: {count}")
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
    print(f"\n⚠️  Batch API는 최대 24시간 소요될 수 있습니다.")

    # 확인
    confirm = input("\n계속 진행하시겠습니까? (y/N): ")
    if confirm.lower() != 'y':
        print("취소되었습니다.")
        return

    # 출력 경로
    output_path = Path(output) if output else Path("data/batch")
    output_path.mkdir(parents=True, exist_ok=True)

    # BatchClient 초기화
    actual_track = "A" if track == "A" else "B"
    client = BatchClient(track=actual_track)

    # 프롬프트 생성
    print("\n⏳ 프롬프트 생성 중...")
    prompts = []
    for cat in categories:
        track_for_cat = CounselingCategory.get_track(cat)
        system_prompt = get_system_prompt(track=track_for_cat, include_thinking=True)

        for i in range(count):
            scenario_prompt = get_scenario_prompt(
                category=cat,
                min_turns=min_turns,
                max_turns=max_turns,
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
            })

    print(f"✅ 프롬프트 생성 완료: {len(prompts)}건")

    # 요청 빌드
    requests = client.build_requests(prompts)

    # 배치 작업 제출
    print("\n⏳ 배치 작업 제출 중...")
    job = await client.create_batch_job(
        requests=requests,
        display_name=f"counseling-{track}-{len(prompts)}",
    )
    print(f"✅ 배치 작업 제출 완료")
    print(f"   - 작업 ID: {job.job_id}")
    print(f"   - 작업 이름: {job.job_name}")
    print(f"   - 상태: {job.state}")

    # 작업 정보 저장
    job_info_path = output_path / f"batch_job_{job.job_id}.json"
    import json
    with open(job_info_path, "w", encoding="utf-8") as f:
        json.dump({
            "job_id": job.job_id,
            "job_name": job.job_name,
            "state": job.state,
            "created_at": job.created_at.isoformat(),
            "total_requests": job.total_requests,
            "track": track,
            "categories": [c.value for c in categories],
            "count_per_category": count,
        }, f, ensure_ascii=False, indent=2)
    print(f"   - 작업 정보 저장: {job_info_path}")

    # 완료 대기
    print(f"\n⏳ 배치 작업 완료 대기 중... (Ctrl+C로 백그라운드 전환)")
    print(f"   (작업이 완료되면 자동으로 결과를 저장합니다)")

    def progress_callback(state: str):
        print(f"   상태: {state}")

    try:
        result = await client.wait_for_completion(
            job,
            poll_interval=poll_interval,
            progress_callback=progress_callback,
        )

        # 결과 저장
        print(f"\n✅ 배치 작업 완료!")
        print(f"   - 성공: {len(result.responses)}건")
        print(f"   - 실패: {len(result.errors)}건")
        print(f"   - 예상 비용: ${result.estimated_cost:.2f}")

        # 응답 저장
        responses_path = output_path / f"responses_{job.job_id}.jsonl"
        with open(responses_path, "w", encoding="utf-8") as f:
            for i, resp in enumerate(result.responses):
                line = json.dumps({
                    "index": i,
                    "category": prompts[i]["category"] if i < len(prompts) else None,
                    "text": resp.get("text", ""),
                }, ensure_ascii=False)
                f.write(line + "\n")
        print(f"   - 응답 저장: {responses_path}")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  백그라운드로 전환됨")
        print(f"   작업 이름: {job.job_name}")
        print(f"   나중에 상태 확인: python main.py --mode batch-status --job-name {job.job_name}")


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
