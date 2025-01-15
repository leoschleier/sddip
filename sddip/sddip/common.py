from enum import StrEnum, auto


class CutType(StrEnum):
    """Types of cuts for SDDiP."""

    BENDERS = auto()
    STRENGTHENED_BENDERS = auto()
    LAGRANGIAN = auto()

    @classmethod
    def from_str(cls, s: str) -> "CutType":
        """Converts a string to a CutType."""
        match s:
            case "BENDERS" | "benders" | "b":
                return cls.BENDERS
            case "STRENGTHENED_BENDERS" | "strengthened_benders" | "sb":
                return cls.STRENGTHENED_BENDERS
            case "LAGRANGIAN" | "lagrangian" | "l":
                return cls.LAGRANGIAN
            case _:
                msg = f"Invalid CutType: {s}"
                raise ValueError(msg)
