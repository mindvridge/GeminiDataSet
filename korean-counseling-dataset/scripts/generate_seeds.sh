#!/bin/bash
# =============================================================================
# Pro Seed 데이터 생성 스크립트
# Track A (고위험군) 카테고리당 1,000개, 총 5,000건 생성
# =============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Pro Seed 데이터 생성 스크립트${NC}"
echo -e "${BLUE}   카테고리당 1,000개, 총 5,000건${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 스크립트 위치 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 1. Python 환경 확인
echo -e "${YELLOW}[1/4] Python 환경 확인...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3가 설치되어 있지 않습니다.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python3: $(python3 --version)${NC}"

# 2. 의존성 설치
echo -e "${YELLOW}[2/4] 의존성 설치...${NC}"
pip install -q pydantic pydantic-settings google-genai python-dotenv

# 3. 인증 확인
echo -e "${YELLOW}[3/4] Google Cloud 인증 확인...${NC}"

# .env 파일에서 설정 읽기
if [ -f ".env" ]; then
    source <(grep -E '^(GOOGLE_GENAI_USE_VERTEXAI|GOOGLE_API_KEY)=' .env | sed 's/^/export /')
fi

if [ "$GOOGLE_GENAI_USE_VERTEXAI" = "True" ] || [ "$GOOGLE_GENAI_USE_VERTEXAI" = "true" ]; then
    echo "Vertex AI 모드 사용"
    # ADC 확인
    if ! gcloud auth application-default print-access-token &> /dev/null; then
        echo -e "${YELLOW}⚠️  ADC 인증이 필요합니다. 다음 명령을 실행하세요:${NC}"
        echo ""
        echo "    gcloud auth application-default login --project=korean-counseling-ai-482108"
        echo ""
        exit 1
    fi
    echo -e "${GREEN}✅ ADC 인증 확인됨${NC}"
else
    echo "AI Studio 모드 사용"
    if [ -z "$GOOGLE_API_KEY" ]; then
        echo -e "${RED}❌ API 키가 설정되지 않았습니다.${NC}"
        echo "    .env 파일에 GOOGLE_API_KEY를 설정하세요."
        exit 1
    fi
    echo -e "${GREEN}✅ API 키 확인됨${NC}"
fi

# 4. 배치 생성 실행
echo -e "${YELLOW}[4/4] Pro Seed 배치 생성 시작...${NC}"
echo ""
echo "📋 설정:"
echo "  - 트랙: A (고위험군)"
echo "  - 카테고리: suicide_crisis, self_harm, domestic_violence, sexual_assault, severe_depression"
echo "  - 카테고리당 생성 수: 1,000건"
echo "  - 총 생성 수: 5,000건"
echo "  - 예상 비용: ~\$169"
echo "  - 예상 소요 시간: 8~10시간"
echo ""

read -p "계속 진행하시겠습니까? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 0
fi

echo ""
echo -e "${GREEN}🚀 배치 생성 시작!${NC}"
echo ""

# 출력 디렉토리 생성
mkdir -p data/seeds/track_a

# 배치 실행
python main.py \
    --mode batch \
    --track A \
    --count 1000 \
    --min-turns 5 \
    --max-turns 10 \
    --output data/seeds/track_a/ \
    --log-level INFO

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   ✅ Pro Seed 생성 완료!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "생성된 파일 위치: data/seeds/track_a/"
echo ""
