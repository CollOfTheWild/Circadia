import json
from pathlib import Path

import pytest

import sleep


def test_load_memories_filters_invalid(tmp_path: Path) -> None:
    mem_path = tmp_path / "memories.jsonl"
    mem_path.write_text(
        '{"messages":[{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}\n'
        'not-json\n'
        '{}\n',
        encoding="utf-8",
    )
    records = sleep.load_memories(mem_path)
    assert len(records) == 1
    assert records[0]["messages"][0]["content"] == "u"


def test_write_sharegpt_roundtrip(tmp_path: Path) -> None:
    data = [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}]
    out_path = sleep.write_sharegpt(data, tmp_path / "train.json")
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded[0]["messages"][1]["content"] == "a"


def test_launch_finetune_builds_cli(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_run(cmd, check):
        called["cmd"] = cmd

    monkeypatch.setattr(sleep.subprocess, "run", fake_run)
    sleep.launch_finetune(
        model_id="mid",
        dataset_path=tmp_path / "train.json",
        adapter_dir=tmp_path / "adapters",
        batch_size=1,
        iters=2,
        learning_rate=1e-4,
        max_seq_len=128,
    )

    cmd = called["cmd"]
    assert "--adapter-path" in cmd


def test_augment_records_parses_json(monkeypatch, tmp_path: Path) -> None:
    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "tmpl"

        def encode(self, text):
            return [0, 1, 2]

    class DummyModel:
        pass

    def fake_load(model_id, adapter_path=None):
        return DummyModel(), DummyTokenizer()

    def fake_stream_generate(*args, **kwargs):
        yield '[{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}]'

    monkeypatch.setattr(sleep, "load", fake_load)
    monkeypatch.setattr(sleep, "stream_generate", fake_stream_generate)

    records = [{"messages": [{"role": "user", "content": "real"}, {"role": "assistant", "content": "reply"}]}]
    new_records, added = sleep.augment_records(
        records=records,
        model_id="mid",
        adapter_dir=tmp_path / "adapters",
        per_record=1,
        max_tokens=64,
        temp=0.7,
        top_p=0.9,
    )

    assert added == 1
    assert len(new_records) == 2
    assert new_records[-1]["messages"][0]["content"] == "x"

