#!/usr/bin/env python3
"""
Elice ML API 연결 테스트

OpenAI 호환 프록시를 통해 Gemini 3 Flash 모델에 접근하는지 확인합니다.
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from generators.openai_client import OpenAICompatibleClient, create_elice_client


async def test_basic_connection():
    """기본 연결 테스트"""
    print("=" * 60)
    print("Elice ML API 연결 테스트")
    print("=" * 60)

    settings = get_settings()

    print(f"\n📋 설정 확인:")
    print(f"  - Base URL: {settings.elice_api_base_url}")
    print(f"  - Model: {settings.elice_model}")
    print(f"  - Track B API: {settings.track_b_api}")

    if not settings.elice_api_base_url or not settings.elice_api_key:
        print("\n❌ Elice API 설정이 없습니다. .env 파일을 확인하세요.")
        return False

    print("\n🔄 클라이언트 초기화 중...")
    client = create_elice_client(settings)

    print(f"\n📡 API 호출 테스트...")
    response = await client.generate(
        system_prompt="당신은 친절한 AI 어시스턴트입니다.",
        user_prompt="안녕하세요! 간단한 인사를 해주세요."
    )

    if response.response_text:
        print(f"\n✅ 연결 성공!")
        print(f"\n📝 응답:")
        print(f"  {response.response_text[:200]}...")

        if response.usage_metadata:
            print(f"\n📊 토큰 사용량:")
            print(f"  - 입력: {response.usage_metadata.get('prompt_tokens', 0)}")
            print(f"  - 출력: {response.usage_metadata.get('response_tokens', 0)}")
            print(f"  - 총계: {response.usage_metadata.get('total_tokens', 0)}")

        return True
    else:
        print(f"\n❌ 연결 실패: 빈 응답")
        return False


async def test_json_generation():
    """JSON 생성 테스트"""
    print("\n" + "=" * 60)
    print("JSON 생성 테스트")
    print("=" * 60)

    settings = get_settings()
    client = create_elice_client(settings)

    system_prompt = """당신은 JSON 형식으로 응답하는 AI입니다.
반드시 아래 형식의 JSON만 출력하세요:
{
    "greeting": "인사말",
    "mood": "기분 상태",
    "advice": "조언"
}"""

    user_prompt = "오늘 하루 힘들었어요. 위로해주세요."

    print("\n🔄 JSON 생성 요청 중...")
    response = await client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    if response.response_text:
        print(f"\n📝 원본 응답:")
        print(response.response_text)

        parsed = client.parse_json_response(response.response_text)
        if parsed:
            print(f"\n✅ JSON 파싱 성공!")
            print(f"  - greeting: {parsed.get('greeting', 'N/A')}")
            print(f"  - mood: {parsed.get('mood', 'N/A')}")
            print(f"  - advice: {parsed.get('advice', 'N/A')}")
            return True
        else:
            print(f"\n⚠️ JSON 파싱 실패")
            return False
    else:
        print(f"\n❌ 응답 없음")
        return False


async def main():
    """메인 테스트 실행"""
    print("\n🚀 Elice ML API 테스트 시작\n")

    # 기본 연결 테스트
    basic_ok = await test_basic_connection()

    if basic_ok:
        # JSON 생성 테스트
        json_ok = await test_json_generation()

    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"  - 기본 연결: {'✅ 성공' if basic_ok else '❌ 실패'}")
    if basic_ok:
        print(f"  - JSON 생성: {'✅ 성공' if json_ok else '❌ 실패'}")

    print("\n✅ 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
