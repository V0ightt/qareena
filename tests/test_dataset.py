"""Dataset builder: labels by construction, unique people across splits, tamper-evident manifests."""

import dataclasses
import datetime as dt
import json

import numpy as np
import pytest

pytest.importorskip("jinja2")  # the builder imports the renderer (optional `synth` group)

from qareena.synth.dataset import SPLITS, seal, sha256, unique_packets, verify, visibility

TODAY = dt.date(2026, 10, 22)


def ink(h=40, w=200):
    m = np.zeros((100, 300), bool)
    m[30 : 30 + h, 50 : 50 + w] = True
    return m


def test_uncovered_ink_is_legible_and_no_ink_is_absent():
    assert visibility(ink(), None)[0] == "legible"
    assert visibility(np.zeros((10, 10), bool), None)[0] == "absent"


def test_one_hidden_character_makes_a_field_unreadable():
    cover = np.zeros((100, 300), np.float32)
    cover[:, 50:80] = 1.0  # about one glyph of a 200 px wide value
    state, total, worst = visibility(ink(), cover)
    assert state == "unreadable"
    assert worst > 0.5
    assert total < 0.2  # little of the field, but a whole character


def test_a_light_partial_cover_is_ambiguous_not_scored():
    cover = np.zeros((100, 300), np.float32)
    cover[30:34, 50:250] = 1.0  # a thin strip across the tops of the letters: 10% of the ink
    assert visibility(ink(), cover)[0] == "ambiguous"


def test_people_are_unique_across_splits_and_reproducible():
    small = {k: dataclasses.replace(v, packets=12) for k, v in SPLITS.items()}
    seen: set[str] = set()
    ids = [
        p.values["iq_national_card_front"]["national_id_number"]
        for s in small.values()
        for p in unique_packets(s, 7, TODAY, seen)
    ]
    assert len(ids) == len(set(ids)) == 12 * len(SPLITS)
    again = [
        p.values["iq_national_card_front"]["national_id_number"]
        for p in unique_packets(small["dev"], 7, TODAY, set())
    ]
    assert again == ids[:12]


def _split(tmp_path):
    root = tmp_path / "dev"
    (root / "dev-0000").mkdir(parents=True)
    f = root / "dev-0000" / "tax_card.png"
    f.write_bytes(b"not really a png")
    (root / "manifest.jsonl").write_text(
        json.dumps({"packet_id": "dev-0000", "files": {"tax_card.png": sha256(f)}}) + "\n", encoding="utf-8"
    )
    record = {
        "split": "dev",
        "packets": 1,
        "seed": 1,
        "today": TODAY.isoformat(),
        "generator_version": "1",
        "generator_commit": "abc",
        "manifest_sha256": sha256(root / "manifest.jsonl"),
        "created": "2026-10-22T00:00:00+00:00",
    }
    (root / "split.json").write_text(json.dumps(record), encoding="utf-8")
    return root, f


def test_manifest_catches_a_changed_or_missing_file(tmp_path):
    root, f = _split(tmp_path)
    assert verify(root) == []
    f.write_bytes(b"edited")
    assert verify(root) == ["changed dev-0000/tax_card.png"]
    f.unlink()
    assert verify(root) == ["missing dev-0000/tax_card.png"]


def test_seal_refuses_a_manifest_edited_after_the_build(tmp_path):
    root, _ = _split(tmp_path)
    path = seal(root, tmp_path / "seals")
    assert json.loads(path.read_text(encoding="utf-8"))["manifest_sha256"] == sha256(root / "manifest.jsonl")
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{}\n")
    with pytest.raises(RuntimeError, match="manifest changed"):
        seal(root, tmp_path / "seals")
