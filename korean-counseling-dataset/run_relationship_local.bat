@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================
echo 🔄 최신 코드 가져오는 중...
echo ==============================================
git pull origin claude/korean-counseling-dataset-Jgp7U

echo.
echo ==============================================
echo 🚀 relationship 생성 시작
echo ==============================================
python run_track_b_safe.py --category relationship --resume --concurrency 10 --skip-long

pause
