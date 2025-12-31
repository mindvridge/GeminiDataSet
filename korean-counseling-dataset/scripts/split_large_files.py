#!/usr/bin/env python3
"""
대용량 JSONL 파일 분할 스크립트

GitHub 100MB 제한을 피하기 위해 큰 파일을 분할합니다.

사용법:
    python scripts/split_large_files.py                    # 자동 검색 & 분할
    python scripts/split_large_files.py --file 파일경로    # 특정 파일 분할
    python scripts/split_large_files.py --max-size 80      # 80MB 기준
"""

import argparse
import os
from pathlib import Path


def get_file_size_mb(file_path: Path) -> float:
    """파일 크기를 MB로 반환"""
    return file_path.stat().st_size / (1024 * 1024)


def split_jsonl_file(file_path: Path, max_size_mb: int = 80) -> list:
    """JSONL 파일을 max_size_mb 이하로 분할"""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"  ❌ 파일 없음: {file_path}")
        return []

    file_size = get_file_size_mb(file_path)
    if file_size <= max_size_mb:
        print(f"  ✓ {file_path.name}: {file_size:.1f}MB (분할 불필요)")
        return []

    print(f"  ✂️ {file_path.name}: {file_size:.1f}MB → 분할 중...")

    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)

    # 예상 파일 수 계산
    num_files = int(file_size / max_size_mb) + 1
    lines_per_file = total_lines // num_files + 1

    # 분할 파일 생성
    created_files = []
    base_name = file_path.stem  # e.g., "medium"
    parent_dir = file_path.parent

    for i in range(num_files):
        start_idx = i * lines_per_file
        end_idx = min((i + 1) * lines_per_file, total_lines)

        if start_idx >= total_lines:
            break

        chunk_lines = lines[start_idx:end_idx]

        # 새 파일명: medium_1.jsonl, medium_2.jsonl, ...
        new_file = parent_dir / f"{base_name}_{i+1}.jsonl"

        with open(new_file, 'w', encoding='utf-8') as f:
            f.writelines(chunk_lines)

        new_size = get_file_size_mb(new_file)
        print(f"     → {new_file.name}: {len(chunk_lines)}건, {new_size:.1f}MB")
        created_files.append(new_file)

    # 원본 파일 삭제
    file_path.unlink()
    print(f"     🗑️ 원본 삭제: {file_path.name}")

    return created_files


def find_and_split_all(base_path: Path, max_size_mb: int = 80):
    """모든 대용량 파일 검색 및 분할"""
    print(f"\n🔍 대용량 파일 검색 중... (기준: {max_size_mb}MB)")
    print(f"   경로: {base_path}\n")

    # 모든 JSONL 파일 검색
    large_files = []
    for jsonl_file in base_path.rglob("*.jsonl"):
        # 이미 분할된 파일 제외 (예: medium_1.jsonl)
        if "_" in jsonl_file.stem and jsonl_file.stem.split("_")[-1].isdigit():
            continue
        # 백업 파일 제외
        if "backup" in jsonl_file.name or "original" in jsonl_file.name:
            continue

        try:
            size = get_file_size_mb(jsonl_file)
            if size > max_size_mb:
                large_files.append((jsonl_file, size))
        except:
            continue

    if not large_files:
        print("✅ 분할이 필요한 파일이 없습니다.")
        return

    print(f"📋 분할 대상: {len(large_files)}개 파일")
    for f, s in large_files:
        print(f"   - {f}: {s:.1f}MB")
    print()

    for file_path, size in large_files:
        print(f"\n[{file_path.parent.name}]")
        split_jsonl_file(file_path, max_size_mb)

    print(f"\n✅ 분할 완료!")


def main():
    parser = argparse.ArgumentParser(description="대용량 JSONL 파일 분할")
    parser.add_argument("--file", "-f", type=str,
                       help="분할할 특정 파일 경로")
    parser.add_argument("--max-size", "-s", type=int, default=80,
                       help="최대 파일 크기 (MB, 기본값: 80)")
    parser.add_argument("--path", "-p", type=str, default="data/raw",
                       help="검색 경로 (기본값: data/raw)")

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("📦 대용량 JSONL 파일 분할 도구")
    print("=" * 50)

    if args.file:
        # 특정 파일 분할
        print(f"\n📄 파일 분할: {args.file}")
        split_jsonl_file(Path(args.file), args.max_size)
    else:
        # 자동 검색 & 분할
        find_and_split_all(Path(args.path), args.max_size)


if __name__ == "__main__":
    main()
