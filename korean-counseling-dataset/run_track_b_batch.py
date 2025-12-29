#!/usr/bin/env python3
"""
Track B 안전 배치 생성 스크립트

100건씩 생성하고 즉시 저장하여 안정성을 높입니다.
중간에 중단되어도 이어서 시작할 수 있습니다.

사용법:
    python run_track_b_batch.py --category career --target 15833
    python run_track_b_batch.py --category career --target 15833 --resume  # 이어서 시작
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


class SafeBatchGenerator:
    """안전한 배치 생성기 - 100건씩 저장"""

    def __init__(
        self,
        category: str,
        target_count: int,
        batch_size: int = 100,
        output_dir: str = "data/raw/track_b",
    ):
        self.category = CounselingCategory(category)
        self.target_count = target_count
        self.batch_size = batch_size
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
        """현재 저장된 데이터 수 확인"""
        file_path = self.output_dir / self.category.value / f"{turn_type}.jsonl"
        if not file_path.exists():
            return 0
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())

    def _save_session(self, session, turn_type: str):
        """세션 즉시 저장 (1건씩)"""
        file_path = self.output_dir / self.category.value / f"{turn_type}.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'a', encoding='utf-8') as f:
            line = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
            f.write(line + "\n")
            f.flush()  # 즉시 디스크에 기록

    async def generate_batch(self, turn_type: str, count: int, generator: GeneralGenerator) -> int:
        """배치 생성 (100건씩)"""
        min_turns, max_turns = self.turn_ranges[turn_type]
        success_count = 0

        for i in range(count):
            try:
                session = await generator.generate_session(
                    category=self.category,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    use_few_shot=True,
                    use_variation=True,
                )

                if session:
                    self._save_session(session, turn_type)
                    success_count += 1
                    self.stats["total_generated"] += 1
                    self.stats["by_type"][turn_type] += 1

                    # 진행률 출력 (10건마다)
                    if (i + 1) % 10 == 0:
                        current = self._get_current_count(turn_type)
                        target = self.distribution[turn_type]
                        logger.info(f"  [{turn_type}] {current}/{target} ({current/target*100:.1f}%)")
                else:
                    self.stats["total_failed"] += 1

            except Exception as e:
                logger.error(f"세션 생성 오류: {e}")
                self.stats["total_failed"] += 1

        return success_count

    async def run(self, resume: bool = False):
        """메인 실행"""
        print("=" * 70)
        print(f"📦 Track B 안전 배치 생성 - {self.category.value}")
        print("=" * 70)

        # 진행 상황 확인
        if resume:
            progress = self._load_progress()
            print(f"\n📋 이전 진행 상황 발견:")
            for t in ["short", "medium", "long"]:
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
        print(f"   - long: {self.distribution['long']}건")
        print(f"\n💾 저장 방식: {self.batch_size}건씩 즉시 저장")
        print(f"📁 저장 위치: {self.output_dir / self.category.value}")

        # Generator 초기화
        generator = GeneralGenerator(settings=self.settings)

        self.stats["start_time"] = datetime.now()

        # 각 턴 타입별 생성
        for turn_type in ["short", "medium", "long"]:
            target = self.distribution[turn_type]
            current = self._get_current_count(turn_type)
            remaining = target - current

            if remaining <= 0:
                print(f"\n✅ [{turn_type}] 이미 완료: {current}/{target}")
                continue

            print(f"\n{'='*70}")
            print(f"📂 [{turn_type}] 생성 시작: {remaining}건 남음 ({current}/{target})")
            print(f"{'='*70}")

            # 100건씩 배치 처리
            batch_num = 0
            while remaining > 0:
                batch_count = min(self.batch_size, remaining)
                batch_num += 1

                print(f"\n🔄 배치 #{batch_num}: {batch_count}건 생성 중...")

                success = await self.generate_batch(turn_type, batch_count, generator)
                remaining -= success

                # 진행 상황 저장
                current = self._get_current_count(turn_type)
                progress[turn_type] = current
                self._save_progress(progress)

                print(f"   ✅ 배치 완료: {success}건 저장 (총 {current}/{target})")

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
        print(f"\n📁 저장 위치: {self.output_dir / self.category.value}")

        # 최종 확인
        print(f"\n📋 최종 데이터:")
        for t in ["short", "medium", "long"]:
            count = self._get_current_count(t)
            target = self.distribution[t]
            status = "✅" if count >= target else "⚠️"
            print(f"   {t}: {count}/{target} {status}")


def main():
    parser = argparse.ArgumentParser(description="Track B 안전 배치 생성")
    parser.add_argument("--category", type=str, required=True,
                       help="카테고리 (career, relationship, academic, mild_anxiety, family_conflict, workplace)")
    parser.add_argument("--target", type=int, default=15833,
                       help="목표 생성 수 (기본값: 15833)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="배치 크기 (기본값: 100)")
    parser.add_argument("--resume", action="store_true",
                       help="이전 작업 이어서 시작")
    parser.add_argument("--output", type=str, default="data/raw/track_b",
                       help="출력 디렉토리")

    args = parser.parse_args()

    # 실행
    generator = SafeBatchGenerator(
        category=args.category,
        target_count=args.target,
        batch_size=args.batch_size,
        output_dir=args.output,
    )

    asyncio.run(generator.run(resume=args.resume))


if __name__ == "__main__":
    main()
