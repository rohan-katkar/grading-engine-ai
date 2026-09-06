import re
from enum import Enum

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"give\s+me\s+(full|10|maximum)\s+(marks|points|score)", re.IGNORECASE),
    re.compile(r"override\s+(the\s+)?rubric", re.IGNORECASE),
    re.compile(r"\[system\s*prompt\]", re.IGNORECASE),
]

PLEA_DETECTION_PATTERNS = [
    re.compile(r"please\s+(give|grant|pass|mark|award)\s+(me|full|points|marks)", re.IGNORECASE),
    re.compile(r"i\s+(really\s+)?need\s+this\s+(grade|mark|pass|class|course)", re.IGNORECASE),
    re.compile(r"my\s+gpa\s+depends\s+on\s+this", re.IGNORECASE),
    re.compile(r"i\s+beg\s+you", re.IGNORECASE),
    re.compile(r"don['']?t\s+fail\s+me", re.IGNORECASE),
    re.compile(r"i\s+studied\s+so\s+hard", re.IGNORECASE),
]

class UserRole(str, Enum):
    STUDENT = "STUDENT"
    EXAM_REVIEWER = "EXAM_REVIEWER"
    EXAM_CREATOR = "EXAM_CREATOR"
    ADMIN = "ADMIN"

    @classmethod
    def list_roles(cls) -> list[str]:
        return [role.value for role in cls]


class QuestionType(str, Enum):
    MCQ = "MCQ"
    LONG_ANSWER = "LONG_ANSWER"
    SHORT_ANSWER = "SHORT_ANSWER"