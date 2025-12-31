import os
from pathlib import Path

print("=== Track B 데이터 확인 ===\n")

base = Path("data/raw/track_b")

if not base.exists():
    print(f"❌ 경로 없음: {base}")
else:
    for cat in ['career','relationship','academic','mild_anxiety','family_conflict','workplace']:
        cat_dir = base / cat
        if cat_dir.exists():
            files = list(cat_dir.glob("*.jsonl"))
            print(f"[{cat}] - {len(files)}개 파일")
            for f in sorted(files):
                size = f.stat().st_size / (1024*1024)
                lines = sum(1 for _ in open(f, encoding='utf-8'))
                print(f"  {f.name}: {lines}건, {size:.1f}MB")
            print()
        else:
            print(f"[{cat}] ❌ 디렉토리 없음\n")
