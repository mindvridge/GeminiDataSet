#!/usr/bin/env python3
"""
Track B 데이터 품질 검사 스크립트

검사 항목:
1. session_id 중복 체크
2. 빈 내용 체크
3. 중복 내용 체크
4. 턴 수 검증
5. 필수 필드 체크
6. 카테고리 일치 확인

사용법:
    python scripts/quality_check.py --category career
    python scripts/quality_check.py --category career --fix  # 문제 자동 수정
"""

import argparse
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4


def load_all_jsonl(category_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """카테고리 디렉토리의 모든 JSONL 파일 로드"""
    all_data = []
    by_type = defaultdict(list)

    for file in sorted(category_dir.glob("*.jsonl")):
        # 백업 파일 제외
        if "backup" in file.name or "original" in file.name:
            continue

        # 타입 추출 (short, medium, long)
        if file.name.startswith("short"):
            turn_type = "short"
        elif file.name.startswith("medium"):
            turn_type = "medium"
        elif file.name.startswith("long"):
            turn_type = "long"
        else:
            continue

        with open(file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        data['_source_file'] = file.name
                        data['_line_num'] = line_num
                        data['_turn_type'] = turn_type
                        all_data.append(data)
                        by_type[turn_type].append(data)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON 파싱 오류: {file.name}:{line_num} - {e}")

    return all_data, dict(by_type)


def check_duplicate_session_ids(data: list[dict]) -> list[dict]:
    """session_id 중복 체크"""
    session_ids = [d.get('session_id', '') for d in data]
    id_counts = Counter(session_ids)
    duplicates = [(sid, cnt) for sid, cnt in id_counts.items() if cnt > 1]

    issues = []
    if duplicates:
        for sid, cnt in sorted(duplicates, key=lambda x: -x[1])[:20]:
            issues.append({
                'type': 'duplicate_session_id',
                'session_id': sid,
                'count': cnt
            })

    return issues


def check_empty_content(data: list[dict]) -> list[dict]:
    """빈 내용 체크"""
    issues = []

    for d in data:
        turns = d.get('turns', [])

        # 턴이 없는 경우
        if not turns:
            issues.append({
                'type': 'no_turns',
                'session_id': d.get('session_id'),
                'source': d.get('_source_file')
            })
            continue

        # 빈 발화 체크
        for i, turn in enumerate(turns):
            client_text = ""
            therapist_text = ""

            if isinstance(turn, dict):
                # 구조 1: turn.client.text, turn.therapist.utterance
                if 'client' in turn and isinstance(turn['client'], dict):
                    client_text = turn['client'].get('text', '')
                if 'therapist' in turn and isinstance(turn['therapist'], dict):
                    therapist_text = turn['therapist'].get('utterance', '')

                # 구조 2: turn.content (role-based)
                if 'content' in turn:
                    if turn.get('role') == 'client':
                        client_text = turn.get('content', '')
                    else:
                        therapist_text = turn.get('content', '')

            if not client_text and not therapist_text:
                issues.append({
                    'type': 'empty_turn',
                    'session_id': d.get('session_id'),
                    'turn_number': i + 1,
                    'source': d.get('_source_file')
                })

    return issues


def check_duplicate_content(data: list[dict]) -> list[dict]:
    """중복 내용 체크 (해시 기반)"""
    content_hashes = {}
    issues = []

    for d in data:
        # 전체 turns를 해시화
        turns = d.get('turns', [])
        content_str = json.dumps(turns, sort_keys=True, ensure_ascii=False)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()

        if content_hash in content_hashes:
            issues.append({
                'type': 'duplicate_content',
                'session_id': d.get('session_id'),
                'duplicate_of': content_hashes[content_hash],
                'source': d.get('_source_file')
            })
        else:
            content_hashes[content_hash] = d.get('session_id')

    return issues


def check_turn_counts(data: list[dict], by_type: dict) -> list[dict]:
    """턴 수 검증"""
    turn_ranges = {
        'short': (5, 10),
        'medium': (15, 25),
        'long': (30, 40)
    }

    issues = []
    stats = defaultdict(lambda: {'valid': 0, 'invalid': 0, 'distribution': Counter()})

    for d in data:
        turn_type = d.get('_turn_type', 'unknown')
        turns = d.get('turns', [])
        turn_count = len(turns)

        stats[turn_type]['distribution'][turn_count] += 1

        if turn_type in turn_ranges:
            min_t, max_t = turn_ranges[turn_type]
            if turn_count < min_t or turn_count > max_t:
                stats[turn_type]['invalid'] += 1
                issues.append({
                    'type': 'invalid_turn_count',
                    'session_id': d.get('session_id'),
                    'turn_type': turn_type,
                    'expected': f"{min_t}-{max_t}",
                    'actual': turn_count,
                    'source': d.get('_source_file')
                })
            else:
                stats[turn_type]['valid'] += 1

    return issues, stats


def check_required_fields(data: list[dict]) -> list[dict]:
    """필수 필드 체크"""
    required = ['session_id', 'category', 'turns']
    issues = []

    for d in data:
        missing = [f for f in required if f not in d or not d[f]]
        if missing:
            issues.append({
                'type': 'missing_fields',
                'session_id': d.get('session_id', 'N/A'),
                'missing': missing,
                'source': d.get('_source_file')
            })

    return issues


def check_category_match(data: list[dict], expected_category: str) -> list[dict]:
    """카테고리 일치 확인"""
    issues = []

    for d in data:
        category = d.get('category', '')
        if category != expected_category:
            issues.append({
                'type': 'category_mismatch',
                'session_id': d.get('session_id'),
                'expected': expected_category,
                'actual': category,
                'source': d.get('_source_file')
            })

    return issues


def fix_duplicate_session_ids(data: list[dict]) -> int:
    """중복 session_id를 UUID로 교체"""
    seen = set()
    fixed = 0

    for d in data:
        sid = d.get('session_id', '')
        if sid in seen or not sid:
            d['session_id'] = str(uuid4())
            fixed += 1
        seen.add(d['session_id'])

    return fixed


def run_quality_check(category: str, base_dir: str = "data/raw/track_b", fix: bool = False):
    """품질 검사 실행"""
    category_dir = Path(base_dir) / category

    if not category_dir.exists():
        print(f"❌ 디렉토리 없음: {category_dir}")
        return

    print("=" * 70)
    print(f"🔍 Track B 품질 검사 - {category}")
    print("=" * 70)

    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    all_data, by_type = load_all_jsonl(category_dir)

    print(f"   총 {len(all_data)}건 로드")
    for t, items in by_type.items():
        print(f"   - {t}: {len(items)}건")

    if not all_data:
        print("❌ 데이터 없음")
        return

    all_issues = []

    # 1. session_id 중복 체크
    print("\n1️⃣ session_id 중복 체크...")
    issues = check_duplicate_session_ids(all_data)
    all_issues.extend(issues)
    if issues:
        print(f"   ⚠️ 중복 발견: {len(issues)}개")
        for issue in issues[:5]:
            print(f"      - {issue['session_id']}: {issue['count']}회")
        if len(issues) > 5:
            print(f"      ... 외 {len(issues) - 5}개")
    else:
        print("   ✅ 중복 없음")

    # 2. 빈 내용 체크
    print("\n2️⃣ 빈 내용 체크...")
    issues = check_empty_content(all_data)
    all_issues.extend(issues)
    if issues:
        print(f"   ⚠️ 빈 내용 발견: {len(issues)}개")
        for issue in issues[:5]:
            print(f"      - {issue['session_id']} ({issue['type']})")
    else:
        print("   ✅ 빈 내용 없음")

    # 3. 중복 내용 체크
    print("\n3️⃣ 중복 내용 체크...")
    issues = check_duplicate_content(all_data)
    all_issues.extend(issues)
    if issues:
        print(f"   ⚠️ 중복 내용 발견: {len(issues)}개")
        for issue in issues[:5]:
            print(f"      - {issue['session_id']} (중복: {issue['duplicate_of']})")
    else:
        print("   ✅ 중복 내용 없음")

    # 4. 턴 수 검증
    print("\n4️⃣ 턴 수 검증...")
    issues, stats = check_turn_counts(all_data, by_type)
    all_issues.extend(issues)

    for turn_type in ['short', 'medium', 'long']:
        if turn_type in stats:
            s = stats[turn_type]
            total = s['valid'] + s['invalid']
            if total > 0:
                pct = s['valid'] / total * 100
                print(f"   [{turn_type}] 유효: {s['valid']}/{total} ({pct:.1f}%)")
                if s['invalid'] > 0:
                    print(f"      ⚠️ 범위 이탈: {s['invalid']}건")
                # 분포 출력
                dist = s['distribution']
                dist_str = ", ".join([f"{k}턴:{v}" for k, v in sorted(dist.items())])
                print(f"      분포: {dist_str}")

    # 5. 필수 필드 체크
    print("\n5️⃣ 필수 필드 체크...")
    issues = check_required_fields(all_data)
    all_issues.extend(issues)
    if issues:
        print(f"   ⚠️ 필드 누락: {len(issues)}개")
    else:
        print("   ✅ 모든 필드 존재")

    # 6. 카테고리 일치 확인
    print("\n6️⃣ 카테고리 일치 확인...")
    issues = check_category_match(all_data, category)
    all_issues.extend(issues)
    if issues:
        print(f"   ⚠️ 카테고리 불일치: {len(issues)}개")
    else:
        print("   ✅ 모든 카테고리 일치")

    # 요약
    print("\n" + "=" * 70)
    print("📊 품질 검사 요약")
    print("=" * 70)
    print(f"   총 데이터: {len(all_data)}건")
    print(f"   문제 발견: {len(all_issues)}건")

    if all_issues:
        issue_types = Counter([i['type'] for i in all_issues])
        print("\n   문제 유형별:")
        for itype, cnt in issue_types.most_common():
            print(f"      - {itype}: {cnt}건")

    # 자동 수정
    if fix and all_issues:
        print("\n🔧 자동 수정 중...")

        # session_id 중복 수정
        fixed = fix_duplicate_session_ids(all_data)
        if fixed > 0:
            print(f"   ✅ session_id 수정: {fixed}건")

            # 파일별로 다시 저장
            # TODO: 분할된 파일 구조 유지하면서 저장
            print("   ⚠️ 파일 저장은 수동으로 진행해주세요")

    print("\n✅ 품질 검사 완료!")

    return all_issues


def main():
    parser = argparse.ArgumentParser(description="Track B 품질 검사")
    parser.add_argument("--category", type=str, required=True,
                       help="검사할 카테고리 (career, relationship 등)")
    parser.add_argument("--fix", action="store_true",
                       help="문제 자동 수정")
    parser.add_argument("--base-dir", type=str, default="data/raw/track_b",
                       help="기본 디렉토리")

    args = parser.parse_args()
    run_quality_check(args.category, args.base_dir, args.fix)


if __name__ == "__main__":
    main()
