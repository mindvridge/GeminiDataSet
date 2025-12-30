#!/usr/bin/env python3
"""
JSONL 파일 분할 스크립트

대용량 JSONL 파일을 GitHub 업로드 가능한 크기로 분할합니다.
GitHub 파일 크기 제한: 100MB

사용법:
    python scripts/split_jsonl.py --input data/raw/track_b/career/short.jsonl --chunk-size 1000
    python scripts/split_jsonl.py --input data/raw/track_b/career/medium.jsonl --chunk-size 500
    python scripts/split_jsonl.py --category career  # 자동 분할
"""

import argparse
import os
from pathlib import Path


def split_jsonl(filepath: str, chunk_size: int) -> list[str]:
    """JSONL 파일을 chunk_size 줄씩 분할"""
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"❌ 파일 없음: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    if total == 0:
        print(f"❌ 빈 파일: {filepath}")
        return []

    # 원본 파일 크기 확인
    original_size = filepath.stat().st_size / (1024 * 1024)  # MB
    print(f"📁 {filepath.name}: {total}줄, {original_size:.1f}MB")

    # 분할 필요 여부 확인
    if original_size < 90:  # 90MB 이하면 분할 불필요
        print(f"   ✅ 분할 불필요 (100MB 미만)")
        return [str(filepath)]

    # 분할
    base = filepath.stem  # 확장자 제외한 파일명
    parent = filepath.parent
    created_files = []

    for i in range(0, total, chunk_size):
        chunk = lines[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        output_file = parent / f"{base}_{chunk_num}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(chunk)

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ {output_file.name}: {len(chunk)}줄, {size_mb:.1f}MB")
        created_files.append(str(output_file))

    # 원본 파일 백업 및 삭제 (선택적)
    backup_file = parent / f"{base}_original_backup.jsonl"
    filepath.rename(backup_file)
    print(f"   📦 원본 백업: {backup_file.name}")

    return created_files


def split_category(category: str, base_dir: str = "data/raw/track_b"):
    """카테고리의 모든 JSONL 파일 분할"""
    category_dir = Path(base_dir) / category

    if not category_dir.exists():
        print(f"❌ 디렉토리 없음: {category_dir}")
        return

    print(f"\n{'='*60}")
    print(f"📂 {category} 카테고리 분할")
    print(f"{'='*60}")

    # 파일별 분할 설정 (턴 수에 따라 파일 크기가 다름)
    settings = {
        "short.jsonl": 1000,   # 5-10턴, 작은 파일
        "medium.jsonl": 300,   # 15-25턴, 큰 파일
        "long.jsonl": 150,     # 30-40턴, 가장 큰 파일
    }

    for filename, chunk_size in settings.items():
        filepath = category_dir / filename
        if filepath.exists():
            split_jsonl(str(filepath), chunk_size)
        else:
            print(f"⚠️ {filename} 없음")


def main():
    parser = argparse.ArgumentParser(description="JSONL 파일 분할")
    parser.add_argument("--input", type=str, help="분할할 JSONL 파일 경로")
    parser.add_argument("--chunk-size", type=int, default=500, help="분할 단위 (줄 수)")
    parser.add_argument("--category", type=str, help="카테고리 전체 분할 (career, relationship 등)")

    args = parser.parse_args()

    if args.category:
        split_category(args.category)
    elif args.input:
        split_jsonl(args.input, args.chunk_size)
    else:
        print("사용법:")
        print("  python scripts/split_jsonl.py --input <파일경로> --chunk-size <줄수>")
        print("  python scripts/split_jsonl.py --category career")


if __name__ == "__main__":
    main()
