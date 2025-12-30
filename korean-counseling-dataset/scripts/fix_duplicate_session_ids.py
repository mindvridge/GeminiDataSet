#!/usr/bin/env python3
"""
session_id 중복 후처리 스크립트

모든 JSONL 파일에서 중복된 session_id를 고유한 UUID로 대체합니다.
원본 session_id는 original_session_id 필드에 보존됩니다.

사용법:
    python scripts/fix_duplicate_session_ids.py --input data/raw/track_b/career --dry-run
    python scripts/fix_duplicate_session_ids.py --input data/raw/track_b/career
    python scripts/fix_duplicate_session_ids.py --input data/raw/track_a
"""

import argparse
import json
import uuid
from pathlib import Path
from collections import Counter
from datetime import datetime


def generate_unique_id(category: str, turn_type: str) -> str:
    """고유한 session_id 생성"""
    short_uuid = str(uuid.uuid4())[:8].upper()
    return f"{category.upper()}-{turn_type.upper()}-{short_uuid}"


def analyze_duplicates(input_path: Path) -> dict:
    """중복 분석"""
    all_ids = []
    file_sessions = {}

    jsonl_files = sorted(input_path.glob("*.jsonl"))

    for file in jsonl_files:
        if "backup" in file.name or "original" in file.name:
            continue

        sessions = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        sessions.append(data)
                        all_ids.append(data.get('session_id', ''))
                    except json.JSONDecodeError:
                        continue
        file_sessions[file] = sessions

    id_counts = Counter(all_ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}

    return {
        'file_sessions': file_sessions,
        'total_sessions': len(all_ids),
        'unique_ids': len(set(all_ids)),
        'duplicate_types': len(duplicates),
        'duplicate_ids': duplicates
    }


def fix_duplicates(input_path: Path, dry_run: bool = False) -> dict:
    """중복 session_id 수정"""
    print(f"\n{'='*60}")
    print(f"session_id 중복 수정: {input_path}")
    print(f"{'='*60}")

    # 카테고리 추출
    category = input_path.name

    # 분석
    analysis = analyze_duplicates(input_path)
    print(f"\n[분석 결과]")
    print(f"  - 총 세션: {analysis['total_sessions']}")
    print(f"  - 고유 ID: {analysis['unique_ids']}")
    print(f"  - 중복 타입: {analysis['duplicate_types']}")

    if analysis['duplicate_types'] == 0:
        print(f"\n✅ 중복 없음!")
        return {'fixed': 0, 'total': analysis['total_sessions']}

    # 전역 ID 추적 (모든 파일에서)
    seen_ids = set()
    fixed_count = 0

    for file, sessions in analysis['file_sessions'].items():
        # 파일명에서 턴 타입 추출
        if "short" in file.name:
            turn_type = "S"
        elif "medium" in file.name:
            turn_type = "M"
        elif "long" in file.name:
            turn_type = "L"
        else:
            turn_type = "X"

        modified = False
        for session in sessions:
            original_id = session.get('session_id', '')

            # 이미 본 ID이거나 중복된 ID인 경우
            if original_id in seen_ids:
                new_id = generate_unique_id(category, turn_type)
                while new_id in seen_ids:
                    new_id = generate_unique_id(category, turn_type)

                if not dry_run:
                    session['original_session_id'] = original_id
                    session['session_id'] = new_id

                seen_ids.add(new_id)
                fixed_count += 1
                modified = True
            else:
                seen_ids.add(original_id)

        # 파일 다시 쓰기
        if modified and not dry_run:
            with open(file, 'w', encoding='utf-8') as f:
                for session in sessions:
                    f.write(json.dumps(session, ensure_ascii=False) + '\n')
            print(f"  ✏️ 수정: {file.name}")

    return {'fixed': fixed_count, 'total': analysis['total_sessions']}


def main():
    parser = argparse.ArgumentParser(description="session_id 중복 수정")
    parser.add_argument("--input", "-i", required=True, help="입력 디렉토리")
    parser.add_argument("--dry-run", action="store_true", help="실제 수정 없이 분석만")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {input_path}")
        return

    print(f"\n🔧 session_id 중복 수정 도구")
    print(f"   입력: {input_path}")
    print(f"   모드: {'분석만 (dry-run)' if args.dry_run else '실제 수정'}")

    result = fix_duplicates(input_path, args.dry_run)

    print(f"\n{'='*60}")
    print(f"[결과]")
    print(f"  - 수정된 세션: {result['fixed']}건")
    print(f"  - 전체 세션: {result['total']}건")

    if args.dry_run:
        print(f"\n⚠️ dry-run 모드: 실제 파일은 수정되지 않았습니다.")
        print(f"   실제 수정하려면 --dry-run 옵션을 제거하세요.")
    else:
        print(f"\n✅ 수정 완료!")


if __name__ == "__main__":
    main()
