#!/usr/bin/env python3
"""
Track B 데이터 후처리 스크립트

1. session_id 중복 → UUID로 교체
2. 데이터 품질 검증
3. 통계 출력

사용법:
    python scripts/postprocess_track_b.py --category career
    python scripts/postprocess_track_b.py --all
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from uuid import uuid4

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "track_b"

CATEGORIES = ["career", "relationship", "academic", "mild_anxiety", "family_conflict", "workplace"]
TURN_TYPES = ["short", "medium", "long"]


def load_jsonl(file_path: Path) -> list[dict]:
    """JSONL 파일 로드"""
    data = []
    if not file_path.exists():
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], file_path: Path):
    """JSONL 파일 저장"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def fix_session_ids(data: list[dict]) -> tuple[list[dict], int]:
    """session_id 중복 수정 → UUID로 교체"""
    seen = set()
    fixed_count = 0

    for item in data:
        old_id = item.get('session_id', '')
        if old_id in seen or not old_id:
            # 중복이거나 비어있으면 새 UUID 생성
            new_id = str(uuid4())
            item['session_id'] = new_id
            fixed_count += 1
        seen.add(item['session_id'])

    return data, fixed_count


def validate_data(data: list[dict], turn_type: str) -> dict:
    """데이터 품질 검증"""
    turn_ranges = {
        "short": (5, 10),
        "medium": (15, 25),
        "long": (30, 40),
    }
    min_turns, max_turns = turn_ranges[turn_type]

    stats = {
        "total": len(data),
        "valid": 0,
        "invalid_turns": 0,
        "missing_fields": 0,
        "turn_distribution": Counter(),
    }

    required_fields = ['session_id', 'category', 'turns']

    for item in data:
        # 필수 필드 확인
        missing = [f for f in required_fields if f not in item]
        if missing:
            stats["missing_fields"] += 1
            continue

        # 턴 수 확인
        turns = len(item.get('turns', []))
        stats["turn_distribution"][turns] += 1

        if turns < min_turns or turns > max_turns:
            stats["invalid_turns"] += 1
        else:
            stats["valid"] += 1

    return stats


def process_category(category: str, dry_run: bool = False):
    """카테고리 처리"""
    print(f"\n{'='*60}")
    print(f"📂 {category} 카테고리 처리")
    print(f"{'='*60}")

    category_dir = DATA_DIR / category
    if not category_dir.exists():
        print(f"   ⚠️ 디렉토리 없음: {category_dir}")
        return

    total_fixed = 0
    total_sessions = 0

    for turn_type in TURN_TYPES:
        file_path = category_dir / f"{turn_type}.jsonl"
        if not file_path.exists():
            print(f"   [{turn_type}] 파일 없음")
            continue

        # 데이터 로드
        data = load_jsonl(file_path)
        if not data:
            print(f"   [{turn_type}] 데이터 없음")
            continue

        print(f"\n   [{turn_type}] {len(data)}건")

        # 검증
        stats = validate_data(data, turn_type)
        print(f"      - 유효: {stats['valid']}건")
        print(f"      - 턴 수 위반: {stats['invalid_turns']}건")
        print(f"      - 필드 누락: {stats['missing_fields']}건")

        # 턴 분포
        print(f"      - 턴 분포: ", end="")
        for t in sorted(stats['turn_distribution'].keys()):
            print(f"{t}턴:{stats['turn_distribution'][t]} ", end="")
        print()

        # session_id 수정
        data, fixed = fix_session_ids(data)
        print(f"      - session_id 수정: {fixed}건")
        total_fixed += fixed
        total_sessions += len(data)

        # 저장
        if not dry_run and fixed > 0:
            save_jsonl(data, file_path)
            print(f"      ✅ 저장 완료")

    print(f"\n   📊 {category} 총계:")
    print(f"      - 총 세션: {total_sessions}건")
    print(f"      - session_id 수정: {total_fixed}건")


def main():
    parser = argparse.ArgumentParser(description="Track B 데이터 후처리")
    parser.add_argument("--category", type=str, help="처리할 카테고리")
    parser.add_argument("--all", action="store_true", help="모든 카테고리 처리")
    parser.add_argument("--dry-run", action="store_true", help="실제 저장 없이 검증만")

    args = parser.parse_args()

    if args.all:
        categories = CATEGORIES
    elif args.category:
        categories = [args.category]
    else:
        print("사용법: --category <name> 또는 --all")
        return

    print("=" * 60)
    print("🔧 Track B 데이터 후처리")
    print("=" * 60)

    if args.dry_run:
        print("⚠️ DRY RUN 모드 - 실제 저장 없음")

    for category in categories:
        process_category(category, dry_run=args.dry_run)

    print("\n✅ 후처리 완료!")


if __name__ == "__main__":
    main()
