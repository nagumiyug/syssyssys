from __future__ import annotations

from dataclasses import dataclass


REQUIRED_COLUMNS = [
    "timestamp",
    "session_id",
    "software",
    "scenario",
    "label",
    "pid",
    "tid",
    "syscall_name",
]


OPTIONAL_COLUMNS = [
    "return_code",
    "duration_us",
    "path",
    "path_type",
    "file_ext",
    "access_mask",
    "op_category",
    "remote_addr",
    "remote_port",
]


@dataclass(frozen=True)
class EventSchema:
    timestamp: str = "timestamp"
    session_id: str = "session_id"
    software: str = "software"
    scenario: str = "scenario"
    label: str = "label"
    pid: str = "pid"
    tid: str = "tid"
    syscall_name: str = "syscall_name"
    return_code: str = "return_code"
    duration_us: str = "duration_us"
    path: str = "path"
    path_type: str = "path_type"
    file_ext: str = "file_ext"
    access_mask: str = "access_mask"
    op_category: str = "op_category"
    remote_addr: str = "remote_addr"
    remote_port: str = "remote_port"


def validate_columns(columns: list[str]) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

