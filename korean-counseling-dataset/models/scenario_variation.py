"""
시나리오 다양성 모듈 (Scenario Variation)

데이터 생성 시 다양한 시나리오와 변수를 조합하여
중복되지 않는 고유한 상담 케이스를 생성합니다.
"""

import json
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from models.schemas import CounselingCategory


@dataclass
class ScenarioVariation:
    """시나리오 변형 정보"""
    scenario_id: str
    title: str
    client_profile: str
    presenting_problem: str
    key_risk_factors: list[str]
    therapeutic_focus: list[str]
    # 변형 요소
    age: int
    gender: str
    occupation: str
    additional_context: str


# ==================== 변형 요소 풀 ====================

AGE_RANGES = {
    "teen": (15, 19, "10대 후반"),
    "young_adult": (20, 29, "20대"),
    "adult": (30, 39, "30대"),
    "middle_aged": (40, 49, "40대"),
    "older_adult": (50, 59, "50대"),
    "senior": (60, 69, "60대"),
}

GENDERS = ["남성", "여성"]

OCCUPATIONS = {
    "student": ["고등학생", "대학생", "대학원생", "취업준비생"],
    "employed": [
        "회사원", "공무원", "교사", "간호사", "의사", "엔지니어",
        "영업사원", "연구원", "디자이너", "프로그래머", "마케터",
        "금융업 종사자", "서비스업 종사자", "자영업자", "프리랜서"
    ],
    "homemaker": ["전업주부", "가사 담당"],
    "unemployed": ["구직자", "실직자", "무직"],
    "retired": ["은퇴자", "퇴직자"],
}

# 카테고리별 추가 컨텍스트
ADDITIONAL_CONTEXTS = {
    CounselingCategory.SUICIDE_CRISIS: [
        "최근 중요한 상실을 경험함",
        "사회적 지지체계가 약함",
        "과거 자살 시도 경험 있음",
        "우울증 병력 있음",
        "최근 생활 스트레스가 급증함",
        "수면 장애로 며칠째 잠을 못 잠",
        "식욕 저하로 체중이 급격히 감소함",
        "일상 활동에 대한 흥미를 완전히 잃음",
    ],
    CounselingCategory.SELF_HARM: [
        "정서 조절에 어려움을 겪어옴",
        "완벽주의적 성향이 강함",
        "가정 내 갈등이 심각함",
        "또래 관계에서 왕따를 경험함",
        "학업/업무 압박이 심함",
        "감정을 표현하는 것이 어려움",
        "과거 외상 경험이 있음",
    ],
    CounselingCategory.DOMESTIC_VIOLENCE: [
        "경제적으로 가해자에게 의존적임",
        "자녀가 있어 떠나기 어려움",
        "가족들이 폭력 사실을 모름",
        "수년간 폭력이 지속되어 왔음",
        "신체적 부상을 입은 적 있음",
        "정서적/언어적 폭력이 주를 이룸",
        "최근 폭력이 심해지고 있음",
    ],
    CounselingCategory.SEXUAL_ASSAULT: [
        "사건 이후 PTSD 증상을 보임",
        "아는 사람에 의한 피해임",
        "사건을 아무에게도 말하지 못함",
        "자기 비난 경향이 강함",
        "수면/악몽 문제가 심각함",
        "회피 행동이 일상에 영향을 줌",
        "친밀감에 대한 두려움이 생김",
    ],
    CounselingCategory.SEVERE_DEPRESSION: [
        "약물 치료 중이나 효과가 미미함",
        "일상 기능이 크게 저하됨",
        "사회적으로 고립되어 있음",
        "만성적인 무기력감을 호소함",
        "과거 우울 삽화가 있었음",
        "가족력이 있음",
        "신체 증상도 동반됨",
    ],
    CounselingCategory.CAREER: [
        "진로 결정에 대한 부모님 압박이 있음",
        "자신의 적성을 잘 모름",
        "여러 분야에 관심이 있어 혼란스러움",
        "취업 준비 과정에서 좌절감을 느낌",
        "전공과 희망 직업이 다름",
        "경제적 상황을 고려해야 함",
    ],
    CounselingCategory.RELATIONSHIP: [
        "연인과의 갈등이 잦음",
        "친구 관계에서 어려움을 겪음",
        "사회적 상황에서 불안함을 느낌",
        "거절에 대한 두려움이 강함",
        "과거 상처로 인해 신뢰가 어려움",
        "의존적인 관계 패턴을 보임",
    ],
    CounselingCategory.ACADEMIC: [
        "성적 하락으로 자신감이 떨어짐",
        "시험 불안이 심함",
        "집중력 저하를 호소함",
        "학업과 다른 활동 사이 균형이 어려움",
        "부모님의 기대가 부담스러움",
        "동기 부여가 되지 않음",
    ],
    CounselingCategory.MILD_ANXIETY: [
        "특정 상황에서 불안이 심해짐",
        "신체 증상(심장 두근거림, 호흡 곤란)을 경험함",
        "걱정이 많아 일상에 영향을 줌",
        "수면에 어려움을 겪음",
        "작은 일에도 긴장함",
        "미래에 대한 불확실성이 두려움",
    ],
    CounselingCategory.FAMILY_CONFLICT: [
        "부모와의 갈등이 심함",
        "형제자매 간 갈등이 있음",
        "세대 차이로 인한 가치관 충돌",
        "부모님의 이혼/별거 상황",
        "가족 내 소통이 단절됨",
        "원가족 문제가 현재에 영향을 줌",
    ],
    CounselingCategory.WORKPLACE: [
        "상사와의 갈등을 겪고 있음",
        "업무 과부하 상태임",
        "직장 내 따돌림을 경험함",
        "이직을 고민 중임",
        "워라밸이 무너진 상태",
        "성과 압박이 심함",
    ],
}


class ScenarioLoader:
    """시나리오 로더 - JSON 파일에서 시나리오 로드 및 변형 생성"""

    def __init__(self, scenarios_dir: Optional[Path] = None):
        """
        Args:
            scenarios_dir: 시나리오 JSON 파일 디렉토리
        """
        if scenarios_dir is None:
            scenarios_dir = Path(__file__).parent.parent / "data" / "scenarios"
        self.scenarios_dir = scenarios_dir
        self._cache: dict = {}
        self._load_scenarios()

    def _load_scenarios(self) -> None:
        """시나리오 파일들 로드"""
        track_a_path = self.scenarios_dir / "track_a_scenarios.json"
        track_b_path = self.scenarios_dir / "track_b_scenarios.json"

        if track_a_path.exists():
            with open(track_a_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for cat_key, cat_data in data.get("categories", {}).items():
                    try:
                        category = CounselingCategory(cat_key)
                        self._cache[category] = cat_data.get("scenarios", [])
                    except ValueError:
                        pass

        if track_b_path.exists():
            with open(track_b_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for cat_key, cat_data in data.get("categories", {}).items():
                    try:
                        category = CounselingCategory(cat_key)
                        self._cache[category] = cat_data.get("scenarios", [])
                    except ValueError:
                        pass

    def get_random_scenario(
        self,
        category: CounselingCategory,
    ) -> Optional[dict]:
        """카테고리에서 랜덤 시나리오 반환"""
        scenarios = self._cache.get(category, [])
        if not scenarios:
            return None
        return random.choice(scenarios)

    def generate_variation(
        self,
        category: CounselingCategory,
        seed: Optional[int] = None,
    ) -> ScenarioVariation:
        """
        다양한 변형 요소를 조합한 시나리오 생성

        Args:
            category: 상담 카테고리
            seed: 랜덤 시드 (재현성 위해)

        Returns:
            ScenarioVariation: 변형된 시나리오
        """
        if seed is not None:
            random.seed(seed)

        # 기본 시나리오 선택
        base_scenario = self.get_random_scenario(category)

        # 연령 선택
        age_key = random.choice(list(AGE_RANGES.keys()))
        age_min, age_max, age_desc = AGE_RANGES[age_key]
        age = random.randint(age_min, age_max)

        # 성별 선택
        gender = random.choice(GENDERS)

        # 직업 선택 (연령에 맞게)
        if age < 20:
            occupation = random.choice(OCCUPATIONS["student"][:2])  # 고등학생, 대학생
        elif age < 25:
            occupation = random.choice(OCCUPATIONS["student"])
        elif age < 65:
            occupation = random.choice(OCCUPATIONS["employed"] + OCCUPATIONS["homemaker"])
        else:
            occupation = random.choice(OCCUPATIONS["retired"])

        # 추가 컨텍스트 선택 (1-2개)
        contexts = ADDITIONAL_CONTEXTS.get(category, [])
        num_contexts = min(random.randint(1, 2), len(contexts))
        selected_contexts = random.sample(contexts, num_contexts) if contexts else []
        additional_context = ", ".join(selected_contexts)

        # 기본 시나리오가 있으면 활용, 없으면 기본값
        if base_scenario:
            # 시나리오의 프로필을 변형
            client_profile = f"{age}세 {gender}, {occupation}"
            presenting_problem = base_scenario.get("presenting_problem", "")
            key_risk_factors = base_scenario.get("key_risk_factors", [])
            therapeutic_focus = base_scenario.get("therapeutic_focus", [])
            scenario_id = base_scenario.get("id", "custom")
            title = base_scenario.get("title", "")
        else:
            client_profile = f"{age}세 {gender}, {occupation}"
            presenting_problem = f"{category.value} 관련 문제를 호소"
            key_risk_factors = []
            therapeutic_focus = []
            scenario_id = "generated"
            title = f"{category.value} 상담"

        return ScenarioVariation(
            scenario_id=scenario_id,
            title=title,
            client_profile=client_profile,
            presenting_problem=presenting_problem,
            key_risk_factors=key_risk_factors,
            therapeutic_focus=therapeutic_focus,
            age=age,
            gender=gender,
            occupation=occupation,
            additional_context=additional_context,
        )


def generate_varied_scenario_prompt(
    category: CounselingCategory,
    min_turns: int = 5,
    max_turns: int = 10,
    seed: Optional[int] = None,
) -> str:
    """
    다양성이 강화된 시나리오 프롬프트 생성

    Args:
        category: 상담 카테고리
        min_turns: 최소 턴 수
        max_turns: 최대 턴 수
        seed: 랜덤 시드

    Returns:
        변형된 시나리오 프롬프트
    """
    loader = ScenarioLoader()
    variation = loader.generate_variation(category, seed)

    # 위험 수준 결정
    risk_mapping = {
        CounselingCategory.SUICIDE_CRISIS: "critical",
        CounselingCategory.SELF_HARM: "critical",
        CounselingCategory.DOMESTIC_VIOLENCE: "high",
        CounselingCategory.SEXUAL_ASSAULT: "high",
        CounselingCategory.SEVERE_DEPRESSION: "high",
        CounselingCategory.CAREER: "low",
        CounselingCategory.RELATIONSHIP: "moderate",
        CounselingCategory.ACADEMIC: "moderate",
        CounselingCategory.MILD_ANXIETY: "moderate",
        CounselingCategory.FAMILY_CONFLICT: "moderate",
        CounselingCategory.WORKPLACE: "moderate",
    }
    risk_level = risk_mapping.get(category, "moderate")

    prompt = f"""## 시나리오: {variation.title}

### 내담자 프로필
- **기본 정보**: {variation.client_profile}
- **주호소**: {variation.presenting_problem}
- **추가 상황**: {variation.additional_context}

### 핵심 위험/문제 요인
{chr(10).join(f"- {factor}" for factor in variation.key_risk_factors) if variation.key_risk_factors else "- 세션 중 탐색 필요"}

### 치료적 초점
{chr(10).join(f"- {focus}" for focus in variation.therapeutic_focus) if variation.therapeutic_focus else "- 내담자 상태에 따라 결정"}

## 생성 지시사항

다음 조건에 맞는 **고유한** 상담 세션을 생성하세요:

1. **카테고리**: {CounselingCategory.get_korean_name(category)}
2. **대화 턴 수**: {min_turns}~{max_turns}턴
3. **위험 수준**: {risk_level}

### 다양성 요구사항
- 위 내담자 프로필을 기반으로 **구체적이고 현실적인** 상담 상황을 생성
- **일반적인 패턴을 피하고** 개인의 고유한 상황과 맥락을 반영
- 내담자의 말투, 표현 방식, 저항 수준 등을 **개성 있게** 표현
- 상담사의 개입도 내담자에 맞춰 **유연하게** 변화

### 필수 요구사항
- 각 턴에서 상담사의 **임상적 판단 과정(clinical_reasoning)**을 상세히 기술
- 내담자의 **감정 변화**를 자연스럽게 반영
- 한국어 **구어체**와 **존댓말** 사용
- 번역투나 부자연스러운 표현 배제

### 세션 구조
1. **도입부 (1-2턴)**: 라포 형성, 주호소 파악
2. **탐색부 (중간)**: 깊은 탐색, 감정 다루기
3. **작업부**: 개입, 통찰 유도
4. **마무리**: 요약, 과제 제시, 다음 회기 예고
"""

    return prompt


# 싱글톤 인스턴스
_scenario_loader: Optional[ScenarioLoader] = None


def get_scenario_loader() -> ScenarioLoader:
    """싱글톤 ScenarioLoader 반환"""
    global _scenario_loader
    if _scenario_loader is None:
        _scenario_loader = ScenarioLoader()
    return _scenario_loader
