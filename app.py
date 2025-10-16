"""Streamlit application for aggregating multi-student course schedules from PDF files.

The app supports uploading multiple timetable PDFs, parsing the course information, and
rendering a combined week view filtered by students. The parsing logic is intentionally
heuristic to accommodate the variety of PDF timetable formats that schools issue.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pdfplumber
import streamlit as st


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


DAY_MAP = {
    "周一": 0,
    "星期一": 0,
    "周二": 1,
    "星期二": 1,
    "周三": 2,
    "星期三": 2,
    "周四": 3,
    "星期四": 3,
    "周五": 4,
    "星期五": 4,
    "周六": 5,
    "星期六": 5,
    "周日": 6,
    "星期日": 6,
    "周天": 6,
    "星期天": 6,
}


DAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


@dataclass
class CourseEntry:
    """Represents a single course slot for a student."""

    student: str
    course_name: str
    location: str
    weekday: int  # 0 = Monday
    time_range: str
    week_start: int
    week_end: int

    def includes_week(self, week: int) -> bool:
        return self.week_start <= week <= self.week_end


# ---------------------------------------------------------------------------
# PDF parsing helpers
# ---------------------------------------------------------------------------


NAME_PATTERNS = [
    re.compile(r"学生[:：]\s*(?P<name>\S+)"),
    re.compile(r"姓名[:：]\s*(?P<name>\S+)"),
    re.compile(r"Student[:：]?\s*(?P<name>[A-Za-z\s]+)"),
]


WEEK_PATTERN = re.compile(r"第\s*(\d+)(?:\s*[-~至到]\s*(\d+))?\s*周")
DAY_PATTERN = re.compile(r"(周[一二三四五六日天]|星期[一二三四五六日天])")
TIME_PATTERN = re.compile(r"(\d{1,2}:\d{2})\s*[-–~至到]\s*(\d{1,2}:\d{2})")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file."""

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        texts = []
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return "\n".join(texts)


def detect_student_name(text: str, fallback: str) -> str:
    for pattern in NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group("name").strip()
    return fallback


def parse_course_lines(text: str) -> Iterable[Tuple[str, str]]:
    """Yield relevant lines that likely contain course information."""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if WEEK_PATTERN.search(line) and DAY_PATTERN.search(line) and TIME_PATTERN.search(line):
            yield raw_line, line


def parse_course_entry(line: str, student: str) -> Optional[CourseEntry]:
    week_match = WEEK_PATTERN.search(line)
    day_match = DAY_PATTERN.search(line)
    time_match = TIME_PATTERN.search(line)

    if not (week_match and day_match and time_match):
        return None

    week_start = int(week_match.group(1))
    week_end = int(week_match.group(2) or week_start)

    day_label = day_match.group(1)
    weekday = DAY_MAP.get(day_label, None)
    if weekday is None:
        return None

    time_range = f"{time_match.group(1)}-{time_match.group(2)}"

    # Remove identified tokens to isolate course name and location information.
    remaining = line
    for segment in (week_match.group(0), day_label, time_match.group(0)):
        remaining = remaining.replace(segment, " ")
    remaining = re.sub(r"\s+", " ", remaining).strip()

    if not remaining:
        course_name = "课程"
        location = ""
    else:
        # Heuristic: treat last token containing digits or building keywords as location.
        tokens = remaining.split(" ")
        if len(tokens) > 1 and re.search(r"\d|教室|楼", tokens[-1]):
            location = tokens[-1]
            course_name = " ".join(tokens[:-1])
        else:
            course_name = remaining
            location = ""

    return CourseEntry(
        student=student,
        course_name=course_name or "课程",
        location=location,
        weekday=weekday,
        time_range=time_range,
        week_start=week_start,
        week_end=week_end,
    )


def parse_pdf_schedule(file_bytes: bytes, fallback_name: str) -> List[CourseEntry]:
    text = extract_text_from_pdf(file_bytes)
    student_name = detect_student_name(text, fallback=fallback_name)

    entries: List[CourseEntry] = []
    for _raw_line, normalized_line in parse_course_lines(text):
        entry = parse_course_entry(normalized_line, student_name)
        if entry:
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def compute_week_number(semester_start: date, reference_date: date) -> int:
    delta_days = (reference_date - semester_start).days
    if delta_days < 0:
        return 1
    return delta_days // 7 + 1


def build_week_dataframe(entries: Sequence[CourseEntry], week: int) -> pd.DataFrame:
    filtered = [e for e in entries if e.includes_week(week)]
    if not filtered:
        return pd.DataFrame(columns=["时间段", *DAY_NAMES])

    time_slots = sorted({e.time_range for e in filtered})

    table: Dict[str, Dict[str, str]] = {}
    for slot in time_slots:
        table[slot] = {day: "无" for day in DAY_NAMES}

    for entry in filtered:
        day_label = DAY_NAMES[entry.weekday]
        description = f"{entry.student}：{entry.course_name}"
        if entry.location:
            description += f"（{entry.location}）"

        existing = table[entry.time_range].get(day_label)
        if existing and existing != "无":
            table[entry.time_range][day_label] = f"{existing}\n{description}"
        else:
            table[entry.time_range][day_label] = description

    df_rows = []
    for slot in time_slots:
        row = {"时间段": slot}
        row.update(table[slot])
        df_rows.append(row)

    return pd.DataFrame(df_rows)


# ---------------------------------------------------------------------------
# Streamlit app layout
# ---------------------------------------------------------------------------


st.set_page_config(page_title="多人课程表聚合工具", layout="wide")

st.title("多人课程表聚合管理工具")

st.markdown(
    """
    该工具支持导入多份学生课程表 PDF，自动解析课程与周次信息，
    并在同一界面中按周展示多人课程安排。通过下方区域上传 PDF、
    选择要查看的周次与学生即可生成课程表。
    """
)


# Layout containers
upload_col, week_col = st.columns([2, 1])

with upload_col:
    st.header("PDF 上传区")
    uploaded_files = st.file_uploader(
        "拖拽或选择上传一个或多个学生课程表 PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

with week_col:
    st.header("周次选择区")
    semester_start = st.date_input(
        "学期起始日期", value=date(date.today().year, 2, 26)
    )

    week_mode = st.radio("周次选择方式", ["自动识别", "手动输入"], index=0)

    if week_mode == "自动识别":
        reference_date = st.date_input("选择日期", value=date.today())
        current_week = compute_week_number(semester_start, reference_date)
    else:
        current_week = st.number_input("输入周次", min_value=1, value=1, step=1)

    st.markdown(f"**当前展示第 {int(current_week)} 周课程**")


all_entries: List[CourseEntry] = []
student_names: List[str] = []

if uploaded_files:
    with st.spinner("正在解析 PDF..."):
        for file in uploaded_files:
            file_bytes = file.read()
            fallback_name = file.name.rsplit(".", 1)[0]
            try:
                entries = parse_pdf_schedule(file_bytes, fallback_name=fallback_name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"解析 {file.name} 失败：{exc}")
                continue

            if entries:
                all_entries.extend(entries)
                student_names.append(entries[0].student)
            else:
                student_names.append(fallback_name)

    student_names = sorted(set(student_names))

st.header("学生筛选区")

if not uploaded_files:
    st.info("请先上传至少一个 PDF 课程表文件。")
    st.stop()

if not all_entries:
    st.warning("未能从上传的 PDF 中解析出课程信息，请检查文件格式。")
    st.stop()

selected_students = st.multiselect(
    "选择要展示课程的学生",
    options=student_names,
    default=student_names,
)

st.header("课程表展示区")

if not selected_students:
    st.info("请选择至少一名学生以查看课程表。")
    st.stop()

filtered_entries = [entry for entry in all_entries if entry.student in selected_students]

schedule_df = build_week_dataframe(filtered_entries, int(current_week))

if schedule_df.empty:
    st.info("所选学生在该周无课程安排。")
else:
    st.dataframe(schedule_df.set_index("时间段"), use_container_width=True)

