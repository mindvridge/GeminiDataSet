#!/usr/bin/env python3
"""
JSONL → PDF 변환 스크립트 (인쇄/보관용)

사용법:
    python scripts/convert_to_pdf.py --input data/raw/track_b/career/short.jsonl --output exports/career_short.pdf
    python scripts/convert_to_pdf.py --input data/raw/track_b/career --output exports/career_all.pdf --limit 50
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("필요한 패키지를 설치합니다...")
    import subprocess
    subprocess.run(["pip", "install", "reportlab"], check=True)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont


def register_korean_font():
    """한글 폰트 등록"""
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont('Korean', font_path))
                return 'Korean'
            except:
                continue

    # 폰트가 없으면 기본 폰트 사용 (한글이 깨질 수 있음)
    print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    return 'Helvetica'


def load_jsonl_files(input_path: Path, limit: int = None) -> list:
    """JSONL 파일 로드"""
    sessions = []

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.jsonl"))

    for file in files:
        if limit and len(sessions) >= limit:
            break

        # 파일명에서 타입 추출
        if "short" in file.name:
            turn_type = "short"
        elif "medium" in file.name:
            turn_type = "medium"
        elif "long" in file.name:
            turn_type = "long"
        else:
            turn_type = "unknown"

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and len(sessions) >= limit:
                    break
                if line.strip():
                    try:
                        data = json.loads(line)
                        data['_turn_type'] = turn_type
                        sessions.append(data)
                    except json.JSONDecodeError:
                        continue

    return sessions


def create_styles(font_name: str):
    """PDF 스타일 생성"""
    styles = getSampleStyleSheet()

    # 제목 스타일
    styles.add(ParagraphStyle(
        name='KoreanTitle',
        fontName=font_name,
        fontSize=18,
        leading=22,
        spaceAfter=20,
        alignment=1,  # center
    ))

    # 섹션 헤더 스타일
    styles.add(ParagraphStyle(
        name='KoreanHeading',
        fontName=font_name,
        fontSize=14,
        leading=18,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#2E5090'),
    ))

    # 세션 제목 스타일
    styles.add(ParagraphStyle(
        name='SessionTitle',
        fontName=font_name,
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor('#1a5f2a'),
        backColor=colors.HexColor('#e8f5e9'),
    ))

    # 본문 스타일
    styles.add(ParagraphStyle(
        name='KoreanBody',
        fontName=font_name,
        fontSize=10,
        leading=14,
        spaceAfter=6,
    ))

    # 내담자 발화 스타일
    styles.add(ParagraphStyle(
        name='ClientStyle',
        fontName=font_name,
        fontSize=10,
        leading=14,
        leftIndent=10,
        spaceBefore=5,
        spaceAfter=3,
        textColor=colors.HexColor('#c0392b'),
    ))

    # 상담사 발화 스타일
    styles.add(ParagraphStyle(
        name='TherapistStyle',
        fontName=font_name,
        fontSize=10,
        leading=14,
        leftIndent=10,
        spaceBefore=5,
        spaceAfter=8,
        textColor=colors.HexColor('#27ae60'),
    ))

    # 메타정보 스타일
    styles.add(ParagraphStyle(
        name='MetaStyle',
        fontName=font_name,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#7f8c8d'),
    ))

    return styles


def create_cover_page(styles, sessions: list) -> list:
    """표지 페이지 생성"""
    elements = []

    elements.append(Spacer(1, 50*mm))
    elements.append(Paragraph("심리상담 데이터셋", styles['KoreanTitle']))
    elements.append(Spacer(1, 10*mm))
    elements.append(Paragraph("품질 검토용 문서", styles['KoreanHeading']))
    elements.append(Spacer(1, 30*mm))

    # 통계 정보
    by_type = {}
    for s in sessions:
        t = s.get('_turn_type', 'unknown')
        by_type[t] = by_type.get(t, 0) + 1

    info_data = [
        ["생성 일시", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["총 세션 수", f"{len(sessions)}건"],
    ]

    for turn_type, count in sorted(by_type.items()):
        info_data.append([f"  - {turn_type}", f"{count}건"])

    table = Table(info_data, colWidths=[60*mm, 60*mm])
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), styles['KoreanBody'].fontName),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)

    elements.append(PageBreak())
    return elements


def create_toc_page(styles, sessions: list) -> list:
    """목차 페이지 생성"""
    elements = []

    elements.append(Paragraph("목차", styles['KoreanHeading']))
    elements.append(Spacer(1, 10*mm))

    for idx, session in enumerate(sessions, 1):
        session_id = session.get('session_id', f'Session_{idx}')[:30]
        turn_type = session.get('_turn_type', '')
        turn_count = len(session.get('turns', []))
        problem = session.get('presenting_problem', '')[:40]

        toc_line = f"{idx}. [{turn_type}] {session_id} - {problem}... ({turn_count}턴)"
        elements.append(Paragraph(toc_line, styles['KoreanBody']))

    elements.append(PageBreak())
    return elements


def create_session_pages(styles, sessions: list) -> list:
    """세션 상세 페이지 생성"""
    elements = []

    for idx, session in enumerate(sessions, 1):
        session_id = session.get('session_id', f'Session_{idx}')
        turn_type = session.get('_turn_type', '')
        client_profile = session.get('client_profile', '')
        problem = session.get('presenting_problem', '')
        turns = session.get('turns', [])
        summary = session.get('session_summary', '')

        # 세션 헤더
        elements.append(Paragraph(
            f"세션 {idx}: {session_id}",
            styles['SessionTitle']
        ))

        # 메타 정보
        meta_info = f"타입: {turn_type} | 턴 수: {len(turns)}턴"
        elements.append(Paragraph(meta_info, styles['MetaStyle']))
        elements.append(Spacer(1, 3*mm))

        # 내담자 프로필
        if client_profile:
            elements.append(Paragraph(f"<b>내담자 프로필:</b> {client_profile}", styles['KoreanBody']))

        # 주호소
        if problem:
            elements.append(Paragraph(f"<b>주호소:</b> {problem}", styles['KoreanBody']))

        elements.append(Spacer(1, 5*mm))

        # 대화 내용
        elements.append(Paragraph("━━━ 대화 내용 ━━━", styles['KoreanBody']))

        for turn_idx, turn in enumerate(turns, 1):
            client = turn.get('client', {})
            therapist = turn.get('therapist', {})

            # 내담자 발화
            client_text = client.get('text', '')
            emotion = client.get('emotional_state', '')
            elements.append(Paragraph(
                f"<b>[턴 {turn_idx}] 내담자</b> ({emotion})",
                styles['ClientStyle']
            ))
            elements.append(Paragraph(client_text, styles['KoreanBody']))

            # 상담사 발화
            utterance = therapist.get('utterance', '')
            technique = therapist.get('empathy_technique', '')
            elements.append(Paragraph(
                f"<b>[턴 {turn_idx}] 상담사</b> ({technique})",
                styles['TherapistStyle']
            ))
            elements.append(Paragraph(utterance, styles['KoreanBody']))
            elements.append(Spacer(1, 3*mm))

        # 세션 요약
        if summary:
            elements.append(Spacer(1, 5*mm))
            elements.append(Paragraph(f"<b>세션 요약:</b> {summary}", styles['KoreanBody']))

        elements.append(PageBreak())

    return elements


def convert_to_pdf(input_path: str, output_path: str, limit: int = None):
    """메인 변환 함수"""
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"📂 데이터 로드 중: {input_path}")
    sessions = load_jsonl_files(input_path, limit)
    print(f"   총 {len(sessions)}건 로드")

    if not sessions:
        print("❌ 로드된 세션이 없습니다.")
        return

    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 한글 폰트 등록
    print("🔤 폰트 설정 중...")
    font_name = register_korean_font()

    # 스타일 생성
    styles = create_styles(font_name)

    # PDF 문서 생성
    print("📄 PDF 생성 중...")
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )

    elements = []

    # 표지
    elements.extend(create_cover_page(styles, sessions))

    # 목차 (50건 이하일 때만)
    if len(sessions) <= 50:
        elements.extend(create_toc_page(styles, sessions))

    # 세션 상세
    elements.extend(create_session_pages(styles, sessions))

    # PDF 빌드
    doc.build(elements)

    print(f"\n✅ PDF 파일 저장 완료: {output_path}")
    print(f"   - 총 {len(sessions)}개 세션 포함")


def main():
    parser = argparse.ArgumentParser(description="JSONL → PDF 변환 (인쇄/보관용)")
    parser.add_argument("--input", "-i", required=True, help="입력 JSONL 파일 또는 디렉토리")
    parser.add_argument("--output", "-o", required=True, help="출력 PDF 파일 경로")
    parser.add_argument("--limit", "-l", type=int, default=None, help="변환할 세션 수 제한 (기본값: 전체)")

    args = parser.parse_args()
    convert_to_pdf(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
