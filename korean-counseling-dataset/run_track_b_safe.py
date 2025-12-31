#!/usr/bin/env python3
"""
Track B 안전한 배치 생성 스크립트

안전성 강화:
- 각 세션 생성 후 즉시 디스크에 저장 (fsync)
- 10건마다 진행 상황 저장
- 100건마다 백업 생성
- 중단 시 복구 가능

사용법:
    python run_track_b_safe.py --category career --target 15833 --resume --concurrency 5
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
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


class SafeBatchGenerator:
    """안전한 배치 생성기"""

    def __init__(
        self,
        category: str,
        target_count: int,
        concurrency: int = 5,
        output_dir: str = "data/raw/track_b",
        skip_long: bool = False,
    ):
        self.category = CounselingCategory(category)
        self.target_count = target_count
        self.concurrency = concurrency
        self.output_dir = Path(output_dir)
        self.skip_long = skip_long
        self.settings = get_settings()

        # 분포 계산
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

    def _load_progress(self) -> dict:
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"short": 0, "medium": 0, "long": 0, "completed": False}

    def _save_progress_sync(self, progress: dict):
        """동기 방식으로 진행 상황 저장 (fsync 포함)"""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def _get_current_count(self, turn_type: str) -> int:
        category_dir = self.output_dir / self.category.value
        count = 0

        single_file = category_dir / f"{turn_type}.jsonl"
        if single_file.exists():
            with open(single_file, 'r', encoding='utf-8') as f:
                count += sum(1 for line in f if line.strip())

        for split_file in sorted(category_dir.glob(f"{turn_type}_*.jsonl")):
            if "backup" in split_file.name:
                continue
            with open(split_file, 'r', encoding='utf-8') as f:
                count += sum(1 for line in f if line.strip())

        return count

    def _create_backup(self, turn_type: str):
        """백업 생성"""
        category_dir = self.output_dir / self.category.value
        backup_dir = category_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for f in category_dir.glob(f"{turn_type}*.jsonl"):
            if "backup" in f.name:
                continue
            backup_file = backup_dir / f"{f.stem}_{timestamp}.jsonl"
            shutil.copy2(f, backup_file)

        logger.info(f"📦 백업 생성: {backup_dir}")

    async def _save_session_safe(self, session, turn_type: str):
        """안전하게 세션 저장 (fsync 포함)"""
        file_path = self.output_dir / self.category.value / f"{turn_type}.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with self._file_lock:
            with open(file_path, 'a', encoding='utf-8') as f:
                line = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())  # 디스크에 강제 기록

            self.session_count += 1

            # 10건마다 진행 상황 저장
            if self.session_count % 10 == 0:
                progress = self._load_progress()
                progress[turn_type] = self._get_current_count(turn_type)
                self._save_progress_sync(progress)

            # 100건마다 백업
            if self.session_count % 100 == 0:
                self._create_backup(turn_type)

    async def _generate_one(
        self,
        generator: GeneralGenerator,
        turn_type: str,
        semaphore: asyncio.Semaphore,
        idx: int,
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
                    logger.info(f"✅ [{idx}] 세션 저장 완료: {session.session_id}")
                    return True
                else:
                    logger.warning(f"⚠️ [{idx}] 빈 세션")
                    return False

            except Exception as e:
                logger.error(f"❌ [{idx}] 오류: {e}")
                return False

    async def run(self, resume: bool = False):
        print("=" * 60)
        print(f"🔒 Track B 안전한 배치 생성 - {self.category.value}")
        print(f"⚡ 동시 요청: {self.concurrency}개")
        print(f"💾 저장 방식: 즉시 저장 + fsync")
        print(f"📦 백업: 100건마다 자동 백업")
        if self.skip_long:
            print(f"⏭️ long 세션 제외")
        print("=" * 60)

        turn_types = ["short", "medium"] if self.skip_long else ["short", "medium", "long"]

        if resume:
            print(f"\n📋 이전 진행 상황:")
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
            print(f"📂 [{turn_type}] 시작: {remaining}건 남음 ({current}/{target})")
            print(f"{'='*60}")

            # 초기 백업
            self._create_backup(turn_type)

            semaphore = asyncio.Semaphore(self.concurrency)
            batch_size = 50  # 작은 배치로 나눔

            while remaining > 0:
                batch_count = min(batch_size, remaining)
                start_idx = current + 1

                print(f"\n🔄 {batch_count}건 생성 중...")

                tasks = [
                    self._generate_one(generator, turn_type, semaphore, start_idx + i)
                    for i in range(batch_count)
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)
                success = sum(1 for r in results if r is True)

                current = self._get_current_count(turn_type)
                remaining = target - current

                # 진행 상황 저장
                progress = self._load_progress()
                progress[turn_type] = current
                self._save_progress_sync(progress)

                print(f"   ✅ {success}건 성공, 현재: {current}/{target}")

        print(f"\n{'='*60}")
        print(f"🎉 생성 완료!")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Track B 안전한 배치 생성")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--target", type=int, default=15833)
    parser.add_argument("--concurrency", type=int, default=5)
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
