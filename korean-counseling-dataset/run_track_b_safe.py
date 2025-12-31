#!/usr/bin/env python3
"""
Track B 안전한 배치 생성 스크립트

안전성 + 속도 + 파일 크기 제한:
- 각 세션 즉시 저장 (fsync)
- 파일 50MB 도달 시 자동 분할 (GitHub 100MB 제한 대응)
- 높은 동시성으로 빠른 생성
- 한 카테고리씩 순차 처리

사용법:
    python run_track_b_safe.py --category relationship --resume --concurrency 10 --skip-long
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from generators.general_generator import GeneralGenerator
from models.schemas import CounselingCategory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 파일 크기 제한 (50MB) - GitHub 100MB 제한 대비 여유
MAX_FILE_SIZE_MB = 50


class SafeBatchGenerator:
    """안전한 배치 생성기"""

    def __init__(
        self,
        category: str,
        target_count: int,
        concurrency: int = 10,
        output_dir: str = "data/raw/track_b",
        skip_long: bool = False,
    ):
        self.category = CounselingCategory(category)
        self.target_count = target_count
        self.concurrency = concurrency
        self.output_dir = Path(output_dir)
        self.skip_long = skip_long
        self.settings = get_settings()

        self.distribution = {
            "short": int(target_count * 0.2),
            "medium": int(target_count * 0.6),
            "long": target_count - int(target_count * 0.2) - int(target_count * 0.6),
        }

        self.turn_ranges = {
            "short": (5, 10),
            "medium": (15, 25),
            "long": (30, 40),
        }

        self.progress_file = self.output_dir / self.category.value / "progress.json"
        self._file_lock = asyncio.Lock()
        self.session_count = 0
        self.current_file_num = {}  # 현재 파일 번호

    def _load_progress(self) -> dict:
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"short": 0, "medium": 0, "long": 0, "completed": False}

    def _save_progress_sync(self, progress: dict):
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def _get_current_count(self, turn_type: str) -> int:
        category_dir = self.output_dir / self.category.value
        count = 0

        for f in category_dir.glob(f"{turn_type}*.jsonl"):
            if "backup" in f.name:
                continue
            with open(f, 'r', encoding='utf-8') as file:
                count += sum(1 for line in file if line.strip())

        return count

    def _get_current_file(self, turn_type: str) -> Path:
        """현재 쓸 파일 경로 반환 (크기 초과 시 새 파일)"""
        category_dir = self.output_dir / self.category.value
        category_dir.mkdir(parents=True, exist_ok=True)

        # 현재 파일 번호 초기화
        if turn_type not in self.current_file_num:
            # 기존 파일 중 가장 큰 번호 찾기
            existing = list(category_dir.glob(f"{turn_type}_*.jsonl"))
            if existing:
                nums = []
                for f in existing:
                    try:
                        num = int(f.stem.split('_')[-1])
                        nums.append(num)
                    except:
                        pass
                self.current_file_num[turn_type] = max(nums) if nums else 0
            else:
                self.current_file_num[turn_type] = 0

        # 현재 파일 경로
        if self.current_file_num[turn_type] == 0:
            current_file = category_dir / f"{turn_type}.jsonl"
        else:
            current_file = category_dir / f"{turn_type}_{self.current_file_num[turn_type]}.jsonl"

        # 파일 크기 확인
        if current_file.exists():
            size_mb = current_file.stat().st_size / (1024 * 1024)
            if size_mb >= MAX_FILE_SIZE_MB:
                # 새 파일로 전환
                self.current_file_num[turn_type] += 1
                current_file = category_dir / f"{turn_type}_{self.current_file_num[turn_type]}.jsonl"
                logger.info(f"📁 새 파일 생성: {current_file.name}")

        return current_file

    async def _save_session_safe(self, session, turn_type: str):
        """안전하게 세션 저장"""
        async with self._file_lock:
            file_path = self._get_current_file(turn_type)

            with open(file_path, 'a', encoding='utf-8') as f:
                line = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

            self.session_count += 1

            # 20건마다 진행 상황 저장
            if self.session_count % 20 == 0:
                progress = self._load_progress()
                progress[turn_type] = self._get_current_count(turn_type)
                self._save_progress_sync(progress)
                logger.info(f"💾 진행 저장: {turn_type} = {progress[turn_type]}건")

    async def _generate_one(
        self,
        generator: GeneralGenerator,
        turn_type: str,
        semaphore: asyncio.Semaphore,
    ) -> bool:
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
                    await self._save_session_safe(session, turn_type)
                    return True
                return False

            except Exception as e:
                logger.error(f"오류: {e}")
                return False

    async def run(self, resume: bool = False):
        print("=" * 60)
        print(f"🚀 Track B 생성 - {self.category.value}")
        print(f"⚡ 동시 요청: {self.concurrency}개")
        print(f"💾 파일 크기 제한: {MAX_FILE_SIZE_MB}MB")
        if self.skip_long:
            print(f"⏭️ long 세션 제외")
        print("=" * 60)

        turn_types = ["short", "medium"] if self.skip_long else ["short", "medium", "long"]

        if resume:
            print(f"\n📋 현재 진행 상황:")
            for t in turn_types:
                current = self._get_current_count(t)
                target = self.distribution[t]
                print(f"   {t}: {current}/{target}")

        generator = GeneralGenerator(settings=self.settings)

        for turn_type in turn_types:
            target = self.distribution[turn_type]
            current = self._get_current_count(turn_type)
            remaining = target - current

            if remaining <= 0:
                print(f"\n✅ [{turn_type}] 완료: {current}/{target}")
                continue

            print(f"\n{'='*60}")
            print(f"📂 [{turn_type}] 시작: {remaining}건 ({current}/{target})")
            print(f"{'='*60}")

            semaphore = asyncio.Semaphore(self.concurrency)
            batch_size = 100
            batch_num = 0

            while remaining > 0:
                batch_count = min(batch_size, remaining)
                batch_num += 1
                start_time = datetime.now()

                print(f"\n🔄 배치 #{batch_num}: {batch_count}건 생성 중...")

                tasks = [
                    self._generate_one(generator, turn_type, semaphore)
                    for _ in range(batch_count)
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)
                success = sum(1 for r in results if r is True)

                elapsed = (datetime.now() - start_time).total_seconds()
                current = self._get_current_count(turn_type)
                remaining = target - current

                # 진행 상황 저장
                progress = self._load_progress()
                progress[turn_type] = current
                self._save_progress_sync(progress)

                speed = success / elapsed if elapsed > 0 else 0
                print(f"   ✅ {success}건 저장 ({speed:.1f}건/초)")
                print(f"   📊 진행: {current}/{target} ({remaining}건 남음)")

        print(f"\n{'='*60}")
        print(f"🎉 {self.category.value} 카테고리 완료!")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Track B 안전한 배치 생성")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--target", type=int, default=15833)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--output", type=str, default="data/raw/track_b")

    args = parser.parse_args()

    gen = SafeBatchGenerator(
        category=args.category,
        target_count=args.target,
        concurrency=args.concurrency,
        output_dir=args.output,
        skip_long=args.skip_long,
    )

    asyncio.run(gen.run(resume=args.resume))


if __name__ == "__main__":
    main()
