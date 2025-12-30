#!/usr/bin/env python3
"""
Track B 병렬 배치 생성 스크립트

동시 10개 요청으로 10배 속도 향상.
100건씩 저장하여 안정성 유지.

사용법:
    python run_track_b_parallel.py --category career --target 15833
    python run_track_b_parallel.py --category career --target 15833 --resume
    python run_track_b_parallel.py --category career --target 15833 --concurrency 20
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from generators.general_generator import GeneralGenerator
from models.schemas import CounselingCategory

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ParallelBatchGenerator:
    """병렬 배치 생성기 - 동시 요청으로 속도 향상"""

    def __init__(
        self,
        category: str,
        target_count: int,
        batch_size: int = 100,
        concurrency: int = 10,
        output_dir: str = "data/raw/track_b",
    ):
        self.category = CounselingCategory(category)
        self.target_count = target_count
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.output_dir = Path(output_dir)
        self.settings = get_settings()

        # 분포 계산 (balanced: 20/60/20)
        self.distribution = {
            "short": int(target_count * 0.2),
            "medium": int(target_count * 0.6),
            "long": target_count - int(target_count * 0.2) - int(target_count * 0.6),
        }

        # 턴 수 기준
        self.turn_ranges = {
            "short": (5, 10),
            "medium": (15, 25),
            "long": (30, 40),
        }

        # 진행 상황 파일
        self.progress_file = self.output_dir / self.category.value / "progress.json"

        # 파일 쓰기 락
        self._file_lock = asyncio.Lock()

        # 통계
        self.stats = {
            "total_generated": 0,
            "total_failed": 0,
            "start_time": None,
            "by_type": {"short": 0, "medium": 0, "long": 0},
        }

    def _load_progress(self) -> dict:
        """진행 상황 로드"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"short": 0, "medium": 0, "long": 0, "completed": False}

    def _save_progress(self, progress: dict):
        """진행 상황 저장"""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)

    def _get_current_count(self, turn_type: str) -> int:
        """현재 저장된 데이터 수 확인 (분할 파일 지원)"""
        category_dir = self.output_dir / self.category.value
        count = 0

        # 단일 파일 체크
        single_file = category_dir / f"{turn_type}.jsonl"
        if single_file.exists():
            with open(single_file, 'r', encoding='utf-8') as f:
                count += sum(1 for line in f if line.strip())

        # 분할 파일 체크 (예: medium_1.jsonl, medium_2.jsonl, ...)
        for split_file in sorted(category_dir.glob(f"{turn_type}_*.jsonl")):
            # 백업 파일 제외
            if "backup" in split_file.name or "original" in split_file.name:
                continue
            with open(split_file, 'r', encoding='utf-8') as f:
                count += sum(1 for line in f if line.strip())

        return count

    async def _save_session(self, session, turn_type: str):
        """세션 즉시 저장 (스레드 안전)"""
        file_path = self.output_dir / self.category.value / f"{turn_type}.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with self._file_lock:
            with open(file_path, 'a', encoding='utf-8') as f:
                line = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
                f.write(line + "\n")
                f.flush()

    async def _generate_one(
        self,
        generator: GeneralGenerator,
        turn_type: str,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        """단일 세션 생성 (세마포어로 동시성 제한)"""
        async with semaphore:
            try:
                min_turns, max_turns = self.turn_ranges[turn_type]
                session = await generator.generate_session(
                    category=self.category,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    use_few_shot=True,
                    use_variation=True,
                )

                if session:
                    await self._save_session(session, turn_type)
                    self.stats["total_generated"] += 1
                    self.stats["by_type"][turn_type] += 1
                    return True
                else:
                    self.stats["total_failed"] += 1
                    return False

            except Exception as e:
                logger.error(f"세션 생성 오류: {e}")
                self.stats["total_failed"] += 1
                return False

    async def generate_batch_parallel(
        self,
        turn_type: str,
        count: int,
        generator: GeneralGenerator,
    ) -> int:
        """병렬 배치 생성"""
        semaphore = asyncio.Semaphore(self.concurrency)

        # 모든 작업을 병렬로 생성
        tasks = [
            self._generate_one(generator, turn_type, semaphore)
            for _ in range(count)
        ]

        # 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 성공 카운트
        success_count = sum(1 for r in results if r is True)
        return success_count

    async def run(self, resume: bool = False, skip_long: bool = False):
        """메인 실행"""
        print("=" * 70)
        print(f"🚀 Track B 병렬 배치 생성 - {self.category.value}")
        print(f"⚡ 동시 요청 수: {self.concurrency}개")
        if skip_long:
            print(f"⏭️ long 세션 생성 제외")
        print("=" * 70)

        # 생성할 턴 타입 목록
        turn_types = ["short", "medium"] if skip_long else ["short", "medium", "long"]

        # 진행 상황 확인
        if resume:
            progress = self._load_progress()
            print(f"\n📋 이전 진행 상황 발견:")
            for t in turn_types:
                current = self._get_current_count(t)
                target = self.distribution[t]
                print(f"   {t}: {current}/{target}")
        else:
            progress = {"short": 0, "medium": 0, "long": 0}

        # 목표 출력
        print(f"\n🎯 목표:")
        print(f"   총 {self.target_count}건 (balanced 분포)")
        print(f"   - short: {self.distribution['short']}건")
        print(f"   - medium: {self.distribution['medium']}건")
        if not skip_long:
            print(f"   - long: {self.distribution['long']}건")
        print(f"\n💾 저장 방식: {self.batch_size}건씩 즉시 저장")
        print(f"📁 저장 위치: {self.output_dir / self.category.value}")

        # Generator 초기화
        generator = GeneralGenerator(settings=self.settings)

        self.stats["start_time"] = datetime.now()

        # 각 턴 타입별 생성
        for turn_type in turn_types:
            target = self.distribution[turn_type]
            current = self._get_current_count(turn_type)
            remaining = target - current

            if remaining <= 0:
                print(f"\n✅ [{turn_type}] 이미 완료: {current}/{target}")
                continue

            print(f"\n{'='*70}")
            print(f"📂 [{turn_type}] 생성 시작: {remaining}건 남음 ({current}/{target})")
            print(f"{'='*70}")

            # 배치 처리
            batch_num = 0
            while remaining > 0:
                batch_count = min(self.batch_size, remaining)
                batch_num += 1

                batch_start = datetime.now()
                print(f"\n🔄 배치 #{batch_num}: {batch_count}건 생성 중... (동시 {self.concurrency}개)")

                success = await self.generate_batch_parallel(turn_type, batch_count, generator)
                remaining -= success

                batch_elapsed = (datetime.now() - batch_start).total_seconds()

                # 진행 상황 저장
                current = self._get_current_count(turn_type)
                progress[turn_type] = current
                self._save_progress(progress)

                # 속도 계산
                speed = batch_count / batch_elapsed if batch_elapsed > 0 else 0
                print(f"   ✅ 배치 완료: {success}건 저장 (총 {current}/{target})")
                print(f"   ⏱️ 소요: {batch_elapsed:.1f}초 ({speed:.1f}건/초)")

                # 예상 남은 시간
                total_remaining = sum(
                    max(0, self.distribution[t] - self._get_current_count(t))
                    for t in ["short", "medium", "long"]
                )
                if speed > 0:
                    eta_seconds = total_remaining / speed
                    eta_hours = eta_seconds / 3600
                    print(f"   📊 남은 예상 시간: {eta_hours:.1f}시간 ({total_remaining}건 남음)")

        # 완료
        progress["completed"] = True
        self._save_progress(progress)

        # 최종 통계
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()

        print(f"\n{'='*70}")
        print(f"📊 생성 완료")
        print(f"{'='*70}")
        print(f"   총 생성: {self.stats['total_generated']}건")
        print(f"   실패: {self.stats['total_failed']}건")
        print(f"   소요 시간: {elapsed/60:.1f}분")
        if self.stats['total_generated'] > 0:
            print(f"   평균 속도: {self.stats['total_generated']/elapsed:.1f}건/초")
        print(f"\n📁 저장 위치: {self.output_dir / self.category.value}")

        # 최종 확인
        print(f"\n📋 최종 데이터:")
        for t in ["short", "medium", "long"]:
            count = self._get_current_count(t)
            target = self.distribution[t]
            status = "✅" if count >= target else "⚠️"
            print(f"   {t}: {count}/{target} {status}")


def main():
    parser = argparse.ArgumentParser(description="Track B 병렬 배치 생성")
    parser.add_argument("--category", type=str, required=True,
                       help="카테고리 (career, relationship, academic, mild_anxiety, family_conflict, workplace)")
    parser.add_argument("--target", type=int, default=15833,
                       help="목표 생성 수 (기본값: 15833)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="배치 크기 (기본값: 100)")
    parser.add_argument("--concurrency", type=int, default=10,
                       help="동시 요청 수 (기본값: 10)")
    parser.add_argument("--resume", action="store_true",
                       help="이전 작업 이어서 시작")
    parser.add_argument("--output", type=str, default="data/raw/track_b",
                       help="출력 디렉토리")
    parser.add_argument("--skip-long", action="store_true",
                       help="long 세션 생성 제외 (API 타임아웃 회피)")

    args = parser.parse_args()

    # 실행
    generator = ParallelBatchGenerator(
        category=args.category,
        target_count=args.target,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        output_dir=args.output,
    )

    asyncio.run(generator.run(resume=args.resume, skip_long=args.skip_long))


if __name__ == "__main__":
    main()
