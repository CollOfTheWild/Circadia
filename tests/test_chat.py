import json
from pathlib import Path

from chat import append_memory, detect_adapter


def test_append_memory_writes_jsonl(tmp_path: Path) -> None:
    mem_path = tmp_path / "memories.jsonl"
    append_memory(mem_path, "hello", "world")

    lines = mem_path.read_text(encoding="utf-8").strip().splitlines()
    obj = json.loads(lines[0])
    assert obj["messages"][0]["content"] == "hello"
    assert obj["messages"][1]["content"] == "world"


def test_detect_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "weights").write_text("dummy", encoding="utf-8")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert detect_adapter(adapter_dir) == adapter_dir
    assert detect_adapter(empty_dir) is None

