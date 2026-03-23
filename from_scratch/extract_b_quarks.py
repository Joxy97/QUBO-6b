#!/usr/bin/env python3
"""Extract direct Higgs->bb b-quarks from a HepMC/HepMC.gz file.

This script writes one CSV row per status-23 b/bbar quark that has a unique
ancestor among the three hard-process Higgs bosons in gg -> HHH.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from from_scratch.fastjet import (
    build_higgs_ancestry_resolver,
    find_hard_higgs_labels,
    iter_hepmc2_events,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract direct Higgs->bb b-quarks from a HepMC/HepMC.gz file.",
    )
    parser.add_argument("input", help="Input HepMC file (.hepmc or .hepmc.gz)")
    parser.add_argument(
        "--output",
        default="b_quarks.csv",
        help="Output CSV path (default: b_quarks.csv)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after this many events",
    )
    return parser


def higgs_parent_index(label: str) -> int:
    if not label.startswith("H"):
        raise ValueError(f"Unexpected Higgs label: {label}")
    return int(label[1:])


def run(args: argparse.Namespace) -> int:
    event_count = 0
    bquark_count = 0
    skipped_ambiguous = 0

    with open(args.output, "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(
            [
                "event",
                "hepmc_event",
                "barcode",
                "pdg_id",
                "higgs_parent",
                "pt",
                "eta",
                "y",
                "phi",
                "mass",
                "px",
                "py",
                "pz",
                "E",
            ]
        )

        for i, event in enumerate(iter_hepmc2_events(args.input)):
            if args.max_events is not None and i >= args.max_events:
                break

            hard_higgs = find_hard_higgs_labels(event)
            ancestry = build_higgs_ancestry_resolver(event, hard_higgs)
            event_count += 1

            for barcode in event.particle_order:
                particle = event.particles[barcode]
                if abs(particle.pdg_id) != 5 or particle.status != 23:
                    continue

                labels = ancestry(barcode)
                if len(labels) != 1:
                    if labels:
                        skipped_ambiguous += 1
                    continue

                label = next(iter(labels))
                writer.writerow(
                    [
                        event.event_id,
                        event.hepmc_event_id,
                        barcode,
                        particle.pdg_id,
                        higgs_parent_index(label),
                        f"{particle.vec.pt():.10g}",
                        f"{particle.vec.eta():.10g}",
                        f"{particle.vec.rapidity():.10g}",
                        f"{particle.vec.phi():.10g}",
                        f"{particle.vec.mass():.10g}",
                        f"{particle.vec.px:.10g}",
                        f"{particle.vec.py:.10g}",
                        f"{particle.vec.pz:.10g}",
                        f"{particle.vec.e:.10g}",
                    ]
                )
                bquark_count += 1

    print(
        f"Done. Processed {event_count} events, wrote {bquark_count} b-quarks "
        f"to {args.output}."
    )
    if skipped_ambiguous:
        print(f"Skipped {skipped_ambiguous} ambiguous b-quark entries with multiple Higgs parents.")
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
