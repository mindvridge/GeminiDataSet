#!/bin/bash

# relationship 카테고리 로컬 실행 스크립트

cd "$(dirname "$0")"

echo "=============================================="
echo "🔄 최신 코드 가져오는 중..."
echo "=============================================="
git pull origin claude/korean-counseling-dataset-Jgp7U

echo ""
echo "=============================================="
echo "🚀 relationship 생성 시작"
echo "=============================================="
python run_track_b_safe.py --category relationship --resume --concurrency 10 --skip-long
