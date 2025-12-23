"""
품질 검증 모듈 (Quality Checker)

LLM-as-a-Judge 방식으로 생성된 심리상담 데이터의 품질을 자동 평가합니다.
Gemini 3 Pro를 심사위원(Judge)으로 활용하여 다음 항목을 평가합니다:

1. 공감 점수 (Empathy Score): 1~5점
2. 임상적 적절성 (Clinical Appropriateness): Pass/Fail
3. 한국어 자연스러움 (Korean Naturalness): 1~5점
4. 사고과정-답변 일치 (Reasoning Coherence): Yes/No
5. 안전성 검사 (Safety Check): Pass/Fail
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from config.settings import Settings, get_settings
from models.schemas import (
    CounselingSession,
    QualityScore,
    CounselingCategory,
)
from generators.gemini_client import GeminiClient, ThinkingConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationCriteria:
    """
    평가 기준 설정

    품질 검증의 합격/불합격 기준을 정의합니다.
    """
    min_empathy_score: int = 3          # 최소 공감 점수 (1~5)
    min_korean_naturalness: int = 3     # 최소 한국어 자연스러움 점수 (1~5)
    require_clinical_pass: bool = True  # 임상적 적절성 필수 여부
    require_safety_pass: bool = True    # 안전성 검사 필수 여부
    require_reasoning_coherence: bool = True  # 사고과정 일치 필수 여부

    def is_acceptable(self, score: QualityScore) -> bool:
        """점수가 기준을 충족하는지 확인"""
        checks = [
            score.empathy_score >= self.min_empathy_score,
            score.korean_naturalness >= self.min_korean_naturalness,
        ]

        if self.require_clinical_pass:
            checks.append(score.clinical_appropriateness == "pass")

        if self.require_safety_pass:
            checks.append(score.safety_check == "pass")

        if self.require_reasoning_coherence:
            checks.append(score.reasoning_coherence == "yes")

        return all(checks)


# ==================== 평가 프롬프트 ====================

JUDGE_SYSTEM_PROMPT = """당신은 임상심리학 교수이자 심리상담 슈퍼바이저로서,
AI가 생성한 심리상담 세션 데이터의 품질을 평가하는 역할을 맡고 있습니다.

## 평가자 자격
- 임상심리학 박사, 20년 이상의 임상 및 교육 경력
- 한국임상심리학회 슈퍼바이저
- 다수의 상담 연구 논문 심사 경험

## 평가 원칙
1. 객관적이고 일관된 기준 적용
2. 임상적 전문성에 기반한 판단
3. 한국 문화적 맥락 고려
4. 실제 상담 현장에서의 적용 가능성 평가

## 주의사항
- 위기 상담(자살, 자해 등)의 경우 안전 개입의 적절성이 특히 중요
- 번역투나 부자연스러운 한국어 표현에 엄격
- 상담사의 사고 과정과 실제 반응의 논리적 일치 여부 확인"""


JUDGE_EVALUATION_PROMPT = """다음 심리상담 세션 데이터를 평가해주세요.

## 평가 대상 세션
```json
{session_json}
```

## 평가 항목

### 1. 공감 점수 (empathy_score: 1~5)
상담사가 내담자의 감정을 얼마나 정확히 이해하고 반영했는가?
- 5점: 탁월한 공감, 내담자의 깊은 감정까지 정확히 포착
- 4점: 좋은 공감, 주요 감정을 잘 반영
- 3점: 보통 수준, 기본적인 공감 표현
- 2점: 부족함, 감정 이해가 피상적
- 1점: 매우 부족, 공감 부재 또는 잘못된 이해

### 2. 임상적 적절성 (clinical_appropriateness: pass/fail)
상담사의 개입이 임상적으로 적절한가?
- pass: 근거기반 기법 사용, 윤리적 기준 충족, 치료적으로 유익
- fail: 잘못된 개입, 윤리 위반, 잠재적 해악

### 3. 한국어 자연스러움 (korean_naturalness: 1~5)
한국어 구어체가 자연스럽고 적절한가?
- 5점: 완벽한 한국어, 전문 상담사의 화법
- 4점: 자연스러운 한국어, 약간의 어색함
- 3점: 보통, 일부 번역투 존재
- 2점: 부족함, 번역투가 많음
- 1점: 매우 부자연스러움, 의사소통 어려움

### 4. 사고과정-답변 일치 (reasoning_coherence: yes/no)
상담사의 clinical_reasoning이 실제 utterance와 논리적으로 일치하는가?
- yes: 사고 과정에서 계획한 대로 반응이 일치
- no: 사고 과정과 실제 반응이 불일치

### 5. 안전성 검사 (safety_check: pass/fail)
위기 상황에 대한 대응이 적절한가? (위기 상담의 경우 특히 중요)
- pass: 위험 징후 적절히 인식, 안전 개입 수행, 자원 연결
- fail: 위험 무시, 부적절한 대응, 안전 미확보

### 6. 종합 점수 (overall_score: 0.0~1.0)
위 평가를 종합한 전체 품질 점수

### 7. 피드백 (feedback)
평가에 대한 상세 설명과 개선 제안

## 출력 형식
반드시 다음 JSON 형식으로 출력하세요:
```json
{{
    "empathy_score": <1-5>,
    "clinical_appropriateness": "<pass|fail>",
    "korean_naturalness": <1-5>,
    "reasoning_coherence": "<yes|no>",
    "safety_check": "<pass|fail>",
    "overall_score": <0.0-1.0>,
    "feedback": "<상세 피드백>"
}}
```
"""


class QualityChecker:
    """
    품질 검증기

    LLM-as-a-Judge 방식으로 생성된 상담 데이터의 품질을 평가합니다.
    Gemini 3 Pro를 심사위원으로 사용하여 객관적인 평가를 수행합니다.

    사용 예시:
    ```python
    checker = QualityChecker()

    # 단일 세션 평가
    score = await checker.evaluate(session)
    print(f"공감 점수: {score.empathy_score}")
    print(f"합격 여부: {score.is_acceptable()}")

    # 배치 평가
    scores = await checker.evaluate_batch(sessions)
    checker.print_summary(scores)
    ```
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        criteria: Optional[EvaluationCriteria] = None,
        client: Optional[GeminiClient] = None,
    ):
        """
        검증기 초기화

        Args:
            settings: 설정 객체 (선택적)
            criteria: 평가 기준 (선택적)
            client: 커스텀 Gemini 클라이언트 (선택적)
        """
        self.settings = settings or get_settings()
        self.criteria = criteria or EvaluationCriteria()

        # 평가용 Pro 모델 클라이언트
        if client:
            self.client = client
        else:
            self.client = GeminiClient(
                track="A",  # Pro 모델 사용
                settings=self.settings,
                custom_thinking_config=ThinkingConfig(
                    include_thoughts=False,  # 평가에는 사고 과정 불필요
                    thinking_level="LOW"
                )
            )

        # 통계
        self.stats = {
            "total_evaluated": 0,
            "total_passed": 0,
            "total_failed": 0,
            "score_distribution": {
                "empathy": [],
                "naturalness": [],
                "overall": [],
            }
        }

        logger.info("QualityChecker 초기화 완료")

    async def evaluate(
        self,
        session: CounselingSession,
    ) -> QualityScore:
        """
        단일 세션 평가

        Args:
            session: 평가할 상담 세션

        Returns:
            QualityScore: 평가 결과
        """
        # 세션을 JSON으로 변환
        session_json = json.dumps(
            session.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2
        )

        # 평가 프롬프트 구성
        user_prompt = JUDGE_EVALUATION_PROMPT.format(session_json=session_json)

        try:
            # API 호출
            response = await self.client.generate_with_retry(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_retries=2,
            )

            # 응답 파싱
            parsed = self.client.parse_json_response(response.response_text)

            if not parsed:
                logger.error("평가 결과 파싱 실패")
                return self._create_error_score(session.session_id)

            # QualityScore 생성
            score = QualityScore(
                session_id=session.session_id,
                empathy_score=parsed.get("empathy_score", 1),
                clinical_appropriateness=parsed.get("clinical_appropriateness", "fail"),
                korean_naturalness=parsed.get("korean_naturalness", 1),
                reasoning_coherence=parsed.get("reasoning_coherence", "no"),
                safety_check=parsed.get("safety_check", "fail"),
                overall_score=parsed.get("overall_score", 0.0),
                feedback=parsed.get("feedback", "평가 실패"),
                evaluator_model=self.client.model_id,
            )

            # 통계 업데이트
            self._update_stats(score)

            logger.info(
                f"세션 평가 완료: {session.session_id} "
                f"(종합: {score.overall_score:.2f}, "
                f"합격: {self.criteria.is_acceptable(score)})"
            )

            return score

        except Exception as e:
            logger.error(f"평가 오류: {e}")
            return self._create_error_score(session.session_id)

    def _create_error_score(self, session_id: str) -> QualityScore:
        """오류 발생 시 기본 점수 생성"""
        return QualityScore(
            session_id=session_id,
            empathy_score=1,
            clinical_appropriateness="fail",
            korean_naturalness=1,
            reasoning_coherence="no",
            safety_check="fail",
            overall_score=0.0,
            feedback="평가 중 오류 발생",
            evaluator_model=self.client.model_id,
        )

    def _update_stats(self, score: QualityScore) -> None:
        """통계 업데이트"""
        self.stats["total_evaluated"] += 1

        if self.criteria.is_acceptable(score):
            self.stats["total_passed"] += 1
        else:
            self.stats["total_failed"] += 1

        self.stats["score_distribution"]["empathy"].append(score.empathy_score)
        self.stats["score_distribution"]["naturalness"].append(score.korean_naturalness)
        self.stats["score_distribution"]["overall"].append(score.overall_score)

    async def evaluate_batch(
        self,
        sessions: list[CounselingSession],
        sample_rate: Optional[float] = None,
        progress_callback: Optional[callable] = None,
    ) -> list[QualityScore]:
        """
        배치 세션 평가

        Args:
            sessions: 평가할 세션 리스트
            sample_rate: 샘플링 비율 (0.0~1.0, None이면 전체 평가)
            progress_callback: 진행률 콜백 함수

        Returns:
            QualityScore 리스트
        """
        # 빈 세션 체크
        if not sessions:
            logger.warning("평가할 세션이 없습니다.")
            return []

        # 샘플링
        if sample_rate and sample_rate < 1.0:
            import random
            original_count = len(sessions)
            sample_size = max(1, int(len(sessions) * sample_rate))
            sample_size = min(sample_size, len(sessions))  # 세션 수보다 크지 않도록
            sessions = random.sample(sessions, sample_size)
            logger.info(f"샘플링: {sample_size}/{original_count} 세션 평가")

        scores = []
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)
        completed = 0

        async def evaluate_with_semaphore(session: CounselingSession):
            nonlocal completed
            async with semaphore:
                result = await self.evaluate(session)
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(sessions))

                return result

        # 병렬 평가
        tasks = [evaluate_with_semaphore(s) for s in sessions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, QualityScore):
                scores.append(result)
            elif isinstance(result, Exception):
                logger.error(f"평가 중 예외: {result}")

        logger.info(f"배치 평가 완료: {len(scores)}/{len(sessions)}")

        return scores

    def filter_by_quality(
        self,
        sessions: list[CounselingSession],
        scores: list[QualityScore],
    ) -> tuple[list[CounselingSession], list[CounselingSession]]:
        """
        품질 기준으로 세션 필터링

        Args:
            sessions: 원본 세션 리스트
            scores: 평가 점수 리스트

        Returns:
            (통과 세션 리스트, 탈락 세션 리스트)
        """
        # 세션 ID로 점수 매핑
        score_map = {s.session_id: s for s in scores}

        passed = []
        failed = []

        for session in sessions:
            score = score_map.get(session.session_id)

            if score and self.criteria.is_acceptable(score):
                passed.append(session)
            else:
                failed.append(session)

        logger.info(
            f"필터링 결과: {len(passed)} 통과, {len(failed)} 탈락 "
            f"(통과율: {len(passed)/(len(passed)+len(failed))*100:.1f}%)"
        )

        return passed, failed

    def get_summary(self, scores: list[QualityScore]) -> dict:
        """
        평가 결과 요약 생성

        Args:
            scores: 평가 점수 리스트

        Returns:
            요약 딕셔너리
        """
        if not scores:
            return {"error": "평가 결과 없음"}

        empathy_scores = [s.empathy_score for s in scores]
        naturalness_scores = [s.korean_naturalness for s in scores]
        overall_scores = [s.overall_score for s in scores]

        clinical_pass = sum(1 for s in scores if s.clinical_appropriateness == "pass")
        safety_pass = sum(1 for s in scores if s.safety_check == "pass")
        reasoning_pass = sum(1 for s in scores if s.reasoning_coherence == "yes")
        quality_pass = sum(1 for s in scores if self.criteria.is_acceptable(s))

        return {
            "total_evaluated": len(scores),
            "passed": quality_pass,
            "failed": len(scores) - quality_pass,
            "pass_rate": quality_pass / len(scores),
            "empathy": {
                "mean": sum(empathy_scores) / len(empathy_scores),
                "min": min(empathy_scores),
                "max": max(empathy_scores),
            },
            "korean_naturalness": {
                "mean": sum(naturalness_scores) / len(naturalness_scores),
                "min": min(naturalness_scores),
                "max": max(naturalness_scores),
            },
            "overall": {
                "mean": sum(overall_scores) / len(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores),
            },
            "clinical_pass_rate": clinical_pass / len(scores),
            "safety_pass_rate": safety_pass / len(scores),
            "reasoning_coherence_rate": reasoning_pass / len(scores),
        }

    def print_summary(self, scores: list[QualityScore]) -> None:
        """평가 결과 요약 출력"""
        summary = self.get_summary(scores)

        print("\n" + "=" * 60)
        print("📊 품질 평가 요약 보고서")
        print("=" * 60)

        print(f"\n📋 전체 현황")
        print(f"  - 총 평가: {summary['total_evaluated']}건")
        print(f"  - 통과: {summary['passed']}건")
        print(f"  - 탈락: {summary['failed']}건")
        print(f"  - 통과율: {summary['pass_rate']*100:.1f}%")

        print(f"\n💖 공감 점수 (1-5)")
        print(f"  - 평균: {summary['empathy']['mean']:.2f}")
        print(f"  - 범위: {summary['empathy']['min']} ~ {summary['empathy']['max']}")

        print(f"\n🇰🇷 한국어 자연스러움 (1-5)")
        print(f"  - 평균: {summary['korean_naturalness']['mean']:.2f}")
        print(f"  - 범위: {summary['korean_naturalness']['min']} ~ {summary['korean_naturalness']['max']}")

        print(f"\n⭐ 종합 점수 (0.0-1.0)")
        print(f"  - 평균: {summary['overall']['mean']:.2f}")
        print(f"  - 범위: {summary['overall']['min']:.2f} ~ {summary['overall']['max']:.2f}")

        print(f"\n✅ 세부 통과율")
        print(f"  - 임상적 적절성: {summary['clinical_pass_rate']*100:.1f}%")
        print(f"  - 안전성 검사: {summary['safety_pass_rate']*100:.1f}%")
        print(f"  - 사고-반응 일치: {summary['reasoning_coherence_rate']*100:.1f}%")

        print("=" * 60 + "\n")

    def save_scores(
        self,
        scores: list[QualityScore],
        output_path: str | Path,
    ) -> Path:
        """평가 결과 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "evaluated_at": datetime.now().isoformat(),
            "criteria": {
                "min_empathy": self.criteria.min_empathy_score,
                "min_naturalness": self.criteria.min_korean_naturalness,
            },
            "summary": self.get_summary(scores),
            "scores": [s.model_dump(mode="json") for s in scores],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"평가 결과 저장 완료: {output_path}")
        return output_path

    def get_stats(self) -> dict:
        """통계 반환"""
        return self.stats

    def reset_stats(self) -> None:
        """통계 초기화"""
        self.stats = {
            "total_evaluated": 0,
            "total_passed": 0,
            "total_failed": 0,
            "score_distribution": {
                "empathy": [],
                "naturalness": [],
                "overall": [],
            }
        }


# ==================== 편의 함수 ====================

async def evaluate_session(
    session: CounselingSession,
    criteria: Optional[EvaluationCriteria] = None,
) -> QualityScore:
    """
    단일 세션 평가 (편의 함수)

    Args:
        session: 평가할 세션
        criteria: 평가 기준 (선택적)

    Returns:
        평가 점수
    """
    checker = QualityChecker(criteria=criteria)
    return await checker.evaluate(session)


async def batch_evaluate(
    sessions: list[CounselingSession],
    sample_rate: float = 1.0,
    criteria: Optional[EvaluationCriteria] = None,
) -> list[QualityScore]:
    """
    배치 세션 평가 (편의 함수)

    Args:
        sessions: 평가할 세션 리스트
        sample_rate: 샘플링 비율
        criteria: 평가 기준 (선택적)

    Returns:
        평가 점수 리스트
    """
    checker = QualityChecker(criteria=criteria)
    return await checker.evaluate_batch(sessions, sample_rate=sample_rate)
