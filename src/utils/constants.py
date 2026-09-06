from enum import Enum

class UserRole(str, Enum):
    STUDENT = "STUDENT"
    EXAM_REVIEWER = "EXAM_REVIEWER"
    EXAM_CREATOR = "EXAM_CREATOR"
    ADMIN = "ADMIN"

    @classmethod
    def list_roles(cls) -> list[str]:
        return [role.value for role in cls]