"""Command line: ``qareena synth-packet | run | status``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

from qareena.config import Settings, load_runtime

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _synth_packet(args: argparse.Namespace) -> int:
    from qareena.synth.packets import make_packet
    from qareena.synth.render import STYLES, Renderer

    rng = random.Random(args.seed)  # noqa: S311 - seeded synthetic data, not cryptography
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    packet = make_packet(args.out.name, args.scenario, rng, today=today)
    args.out.mkdir(parents=True, exist_ok=True)
    boxes = {}
    with Renderer() as r:
        for doc_type, values in packet.values.items():
            style = rng.choice(sorted(STYLES))
            res = r.render(f"{doc_type}_{args.layout}", values, args.out / f"{doc_type}.png", style=style)
            boxes[doc_type] = {
                "layout": args.layout,
                "style": style,
                "size": [res.width, res.height],
                "boxes": res.boxes,
            }
    truth = {
        "packet_id": packet.packet_id,
        "scenario": packet.scenario,
        "expected_outcome": packet.expected_outcome,
        "defects": list(packet.defects),
        "values": packet.values,
        "render": boxes,
        "today": today.isoformat(),
    }
    (args.out / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    print(f"{packet.packet_id}: scenario={packet.scenario} expected={packet.expected_outcome} -> {args.out}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from qareena.pipeline.intake import IntakeError, load_image_bytes
    from qareena.pipeline.run import OllamaVlmReader, Pipeline
    from qareena.readers.ocr import ArabicOcr
    from qareena.readers.ollama import OllamaClient

    runtime = load_runtime(Settings())
    images, problems = {}, {}
    for path in sorted(args.packet_dir.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.stem in runtime.catalog.documents:
            try:
                images[path.stem] = load_image_bytes(path.read_bytes()).image
            except IntakeError as exc:
                problems[path.stem] = exc.code
    today = dt.date.fromisoformat(args.today) if args.today else None
    with OllamaClient(runtime.settings.ollama_url) as client:
        pipe = Pipeline(runtime, OllamaVlmReader(runtime, client), ArabicOcr(), today=today)
        try:
            result = pipe.run(args.packet_dir.name, images)
        finally:
            pipe.close()
    out = args.out or (args.packet_dir / "result.json")
    out.write_text(result.model_dump_json(indent=1), encoding="utf-8")
    _print_summary(result, problems)
    print(f"\ntrace: {out}")
    return 0


def _print_summary(result, problems: dict[str, str]) -> None:
    print(f"\n{result.packet_id}: {result.outcome.value}   reasons: {', '.join(result.reasons) or '-'}")
    for doc_type, code in problems.items():
        print(f"  [{doc_type}] rejected at intake: {code}")
    for doc_type, reason in result.retake.items():
        print(f"  [{doc_type}] retake: {reason}")
    for doc_type, doc in result.documents.items():
        t = doc.timings
        timing = f"vlm {t.get('vlm_s', 0):.1f}s  ocr {t.get('ocr_s', 0):.1f}s"
        print(f"  [{doc_type}] {timing}  reasons: {doc.reasons or '-'}")
        for name, rec in doc.fields.items():
            if rec.band is None or rec.band.value == "accept":
                continue
            src = ",".join(s.value for s in rec.sources) or "-"
            print(f"      {rec.band.value:6s} {name:20s} {rec.raw!s:32.32s} p={rec.p_correct} src={src}")
            print(f"             {rec.reasons}")
    for c in result.cross_checks:
        print(f"  cross-check {c.rule}: {c.result} {c.reasons}")


def _build_dataset(args: argparse.Namespace) -> int:
    import dataclasses

    from qareena.synth.dataset import SPLITS, build_split, unique_packets
    from qareena.synth.render import Renderer

    today = dt.date.fromisoformat(args.today)
    seen: set[str] = set()
    for name, spec in SPLITS.items():  # people of earlier splits are drawn first, never reused
        if name == args.split:
            break
        for _ in unique_packets(spec, args.seed, today, seen):
            pass
    spec = SPLITS[args.split]
    if args.packets:
        spec = dataclasses.replace(spec, packets=args.packets)
    with Renderer() as r:
        record = build_split(spec, args.out, seed=args.seed, today=today, seen=seen, renderer=r)
    print(json.dumps(record, indent=1))
    return 0


def _seal(args: argparse.Namespace) -> int:
    from qareena.synth.dataset import seal, verify

    split_dir = args.data / args.split
    problems = verify(split_dir)
    if problems:
        print("\n".join(problems[:20]))
        return 1
    print(f"sealed: {seal(split_dir)}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    from qareena.synth.dataset import verify

    problems = verify(args.data / args.split)
    print("\n".join(problems[:50]) if problems else f"{args.split}: every file matches its manifest")
    return 1 if problems else 0


def _status(args: argparse.Namespace) -> int:
    from qareena.pipeline.scoring import certificate_path, load_certificate
    from qareena.readers.ollama import OllamaClient, OllamaError

    runtime = load_runtime(Settings())
    print(json.dumps(runtime.fingerprint, indent=1, ensure_ascii=False))
    cert = load_certificate(runtime)
    print(f"certificate: {'valid for this policy+model' if cert else 'none'} ({certificate_path(runtime)})")
    try:
        with OllamaClient(runtime.settings.ollama_url, timeout_s=5, retries=0) as c:
            print(f"ollama {c.version()}; loaded: {[m.get('name') for m in c.running()]}")
    except OllamaError as exc:
        print(f"ollama: unreachable ({exc})")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Arabic output on a legacy Windows console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from qareena.synth.packets import SCENARIOS

    ap = argparse.ArgumentParser(prog="qareena", description="Qareena local-first KYC document agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth-packet", help="render a fictional SPECIMEN packet with ground truth")
    s.add_argument("--scenario", default="clean", choices=SCENARIOS)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--layout", default="a", choices=("a", "b", "c"), help="c is the held-out layout")
    s.add_argument("--today", help="ISO date used for validity (default: today)")
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=_synth_packet)
    r = sub.add_parser("run", help="process a folder of document photos named <doc_type>.png")
    r.add_argument("packet_dir", type=Path)
    r.add_argument("--out", type=Path)
    r.add_argument("--today")
    r.set_defaults(func=_run)
    splits = ("dev", "calib", "test_a", "test_b")
    b = sub.add_parser("build-dataset", help="render, degrade and label one evaluation split")
    b.add_argument("--split", required=True, choices=splits)
    b.add_argument("--packets", type=int, help="override the split size (smoke tests)")
    b.add_argument("--seed", type=int, default=20261022)
    b.add_argument("--today", default="2026-10-22", help="ISO date the packets are valid on")
    b.add_argument("--out", type=Path, default=Path("var/data"))
    b.set_defaults(func=_build_dataset)
    se = sub.add_parser("seal", help="verify a built split and write its seal to docs/data/seals")
    se.add_argument("--split", required=True, choices=splits)
    se.add_argument("--data", type=Path, default=Path("var/data"))
    se.set_defaults(func=_seal)
    ve = sub.add_parser("verify", help="re-hash a built split against its manifest")
    ve.add_argument("--split", required=True, choices=splits)
    ve.add_argument("--data", type=Path, default=Path("var/data"))
    ve.set_defaults(func=_verify)
    st = sub.add_parser("status", help="configuration fingerprint, certification and model server health")
    st.set_defaults(func=_status)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
