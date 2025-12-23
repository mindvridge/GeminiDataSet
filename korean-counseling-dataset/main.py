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
from validators.quality_checker import QualityChecker, EvaluationCriteria
from pipeline.batch_processor import BatchProcessor, BatchJob
from pipeline.orchestrator import PipelineOrchestrator, PipelineConfig


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

    print("\n⏳ 세션 생성 중...")

    session = await generator.generate_session(
        category=cat,
        min_turns=min_turns,
        max_turns=max_turns,
    )

    if session:
        print(f"\n✅ 생성 완료!")
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

        # 저장
        if output:
            output_path = Path(output)
            if actual_track == "A":
                generator.save_session(session, output_path)
            else:
                generator.save_session(session, output_path)
            print(f"\n💾 저장 완료: {output_path}")

        return session
    else:
        print("\n❌ 세션 생성 실패")
        return None


async def run_batch_mode(
    track: str,
    category: Optional[str] = None,
    count: int = 10,
    min_turns: int = 5,
    max_turns: int = 10,
    validate: bool = True,
    output: Optional[str] = None,
) -> None:
    """배치 생성 모드"""
    print("\n" + "=" * 60)
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

    # 배치 처리기 생성
    processor = BatchProcessor()

    # 작업 생성
    job = processor.create_job(
        name=f"배치 생성 - {track}",
        track=track,
        categories=categories,
        count_per_category=count,
        min_turns=min_turns,
        max_turns=max_turns,
        validate=validate,
    )

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
    print("\n⏳ 세션 로드 중...")
    sessions = load_sessions_from_jsonl(input_path)
    print(f"  - 로드된 세션: {len(sessions)}건")

    # 검증
    checker = QualityChecker()

    print("\n⏳ 품질 검증 중...")
    scores = await checker.evaluate_batch(sessions, sample_rate=sample_rate)

    # 결과 출력
    checker.print_summary(scores)

    # 저장
    if output:
        output_path = Path(output)
        checker.save_scores(scores, output_path)
        print(f"\n💾 검증 결과 저장: {output_path}")


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

  # 일반상담 100건 배치 생성
  python main.py --mode batch --track B --count 100 --validate

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
        choices=["single", "batch", "pipeline", "validate", "list-categories"],
        default="single",
        help="실행 모드 (default: single)",
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
