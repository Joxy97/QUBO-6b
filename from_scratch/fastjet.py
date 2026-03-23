#!/usr/bin/env python3
"""Cluster jets from a HepMC/HepMC.gz file with anti-kt.

This script reads HepMC2 ASCII records (as produced by Pythia8 in MG5),
selects final-state particles, and clusters jets using anti-kt with R=0.4
by default.

For each clustered jet, it also estimates the origin fractions from the
three hard-process Higgs bosons in gg -> HHH by tracing constituent ancestry.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


NEUTRINO_PDGS = {12, 14, 16, -12, -14, -16}


@dataclass
class FourVector:
    px: float
    py: float
    pz: float
    e: float

    def pt(self) -> float:
        return math.hypot(self.px, self.py)

    def p(self) -> float:
        return math.sqrt(max(self.px * self.px + self.py * self.py + self.pz * self.pz, 0.0))

    def phi(self) -> float:
        return math.atan2(self.py, self.px)

    def rapidity(self) -> float:
        num = self.e + self.pz
        den = self.e - self.pz
        if num <= 0.0 or den <= 0.0:
            return math.copysign(1.0e6, self.pz if self.pz != 0.0 else 1.0)
        return 0.5 * math.log(num / den)

    def eta(self) -> float:
        p = self.p()
        num = p + self.pz
        den = p - self.pz
        if num <= 0.0 or den <= 0.0:
            return math.copysign(1.0e6, self.pz if self.pz != 0.0 else 1.0)
        return 0.5 * math.log(num / den)

    def mass2(self) -> float:
        return self.e * self.e - (self.px * self.px + self.py * self.py + self.pz * self.pz)

    def mass(self) -> float:
        return math.sqrt(max(self.mass2(), 0.0))

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector(
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
            self.e + other.e,
        )


@dataclass
class ParticleRecord:
    barcode: int
    pdg_id: int
    vec: FourVector
    status: int
    end_vertex: int
    prod_vertex: Optional[int]


@dataclass
class EventRecord:
    event_id: int
    hepmc_event_id: int
    particles: Dict[int, ParticleRecord]
    particle_order: List[int]
    stable_barcodes: List[int]
    incoming_by_vertex: Dict[int, List[int]]


@dataclass
class ClusterNode:
    vec: FourVector
    constituents: List[int]

    def pt(self) -> float:
        return self.vec.pt()

    def eta(self) -> float:
        return self.vec.eta()

    def rapidity(self) -> float:
        return self.vec.rapidity()

    def phi(self) -> float:
        return self.vec.phi()

    def mass(self) -> float:
        return self.vec.mass()

    @property
    def px(self) -> float:
        return self.vec.px

    @property
    def py(self) -> float:
        return self.vec.py

    @property
    def pz(self) -> float:
        return self.vec.pz

    @property
    def e(self) -> float:
        return self.vec.e


@dataclass
class EventResult:
    event_id: int
    hepmc_event_id: int
    n_constituents: int
    jets: List[ClusterNode]
    jet_constituent_counts: List[int]
    h_fracs: List[Tuple[float, float, float]]
    truth_flavors: List[str]
    btags: List[Tuple[int, int, int]]


def delta_phi(phi1: float, phi2: float) -> float:
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi <= -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def delta_r2(y1: float, phi1: float, y2: float, phi2: float) -> float:
    dy = y1 - y2
    dphi = delta_phi(phi1, phi2)
    return dy * dy + dphi * dphi


def open_text_file(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def iter_hepmc2_events(
    path: str,
    include_neutrinos: bool = False,
) -> Iterator[EventRecord]:
    """Yield EventRecord objects from a HepMC2 ASCII file."""

    hepmc_event_number: Optional[int] = None
    event_number: int = -1
    particles: Dict[int, ParticleRecord] = {}
    particle_order: List[int] = []
    stable_barcodes: List[int] = []
    current_vertex: Optional[int] = None

    def flush_event() -> Optional[EventRecord]:
        if hepmc_event_number is None:
            return None
        incoming_by_vertex: Dict[int, List[int]] = {}
        for barcode in particle_order:
            particle = particles[barcode]
            if particle.end_vertex == 0:
                continue
            incoming_by_vertex.setdefault(particle.end_vertex, []).append(barcode)
        return EventRecord(
            event_id=event_number,
            hepmc_event_id=hepmc_event_number,
            particles=particles,
            particle_order=particle_order,
            stable_barcodes=stable_barcodes,
            incoming_by_vertex=incoming_by_vertex,
        )

    with open_text_file(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            head = line[0]
            if head == "E" and line.startswith("E "):
                flushed = flush_event()
                if flushed is not None:
                    yield flushed
                fields = line.split()
                hepmc_event_number = int(fields[1]) if len(fields) > 1 else -1
                event_number += 1
                particles = {}
                particle_order = []
                stable_barcodes = []
                current_vertex = None
                continue

            if head == "V" and line.startswith("V "):
                fields = line.split()
                if len(fields) > 1:
                    try:
                        current_vertex = int(fields[1])
                    except ValueError:
                        current_vertex = None
                continue

            if head != "P" or not line.startswith("P "):
                continue
            if hepmc_event_number is None:
                continue

            fields = line.split()
            # HepMC2 particle layout starts as:
            # P barcode pdg px py pz E m status poltheta polphi end_vtx ...
            if len(fields) < 12:
                continue

            try:
                barcode = int(fields[1])
                pdg_id = int(fields[2])
                px = float(fields[3])
                py = float(fields[4])
                pz = float(fields[5])
                e = float(fields[6])
                status = int(fields[8])
                end_vertex = int(fields[11])
            except ValueError:
                continue

            particle = ParticleRecord(
                barcode=barcode,
                pdg_id=pdg_id,
                vec=FourVector(px=px, py=py, pz=pz, e=e),
                status=status,
                end_vertex=end_vertex,
                prod_vertex=current_vertex,
            )
            particles[barcode] = particle
            particle_order.append(barcode)

            if status != 1:
                continue
            if (not include_neutrinos) and pdg_id in NEUTRINO_PDGS:
                continue

            stable_barcodes.append(barcode)

    flushed = flush_event()
    if flushed is not None:
        yield flushed


def anti_kt_cluster(particles: Sequence[ClusterNode], radius: float) -> List[ClusterNode]:
    """Simple anti-kt implementation with E-scheme recombination."""

    if radius <= 0.0:
        raise ValueError("radius must be > 0")

    active: List[ClusterNode] = list(particles)
    jets: List[ClusterNode] = []
    r2 = radius * radius

    while active:
        n = len(active)
        pts = [p.pt() for p in active]
        ys = [p.vec.rapidity() for p in active]
        phis = [p.vec.phi() for p in active]
        invpt2 = [math.inf if pt <= 0.0 else 1.0 / (pt * pt) for pt in pts]

        best_d = math.inf
        best_pair: Optional[Tuple[int, int]] = None
        best_beam: Optional[int] = None

        for i in range(n):
            dib = invpt2[i]
            if dib < best_d:
                best_d = dib
                best_pair = None
                best_beam = i

        for i in range(n):
            for j in range(i + 1, n):
                dij = min(invpt2[i], invpt2[j]) * delta_r2(ys[i], phis[i], ys[j], phis[j]) / r2
                if dij < best_d:
                    best_d = dij
                    best_pair = (i, j)
                    best_beam = None

        if best_pair is not None:
            i, j = best_pair
            merged = ClusterNode(
                vec=active[i].vec + active[j].vec,
                constituents=active[i].constituents + active[j].constituents,
            )
            if i > j:
                i, j = j, i
            active.pop(j)
            active.pop(i)
            active.append(merged)
        else:
            assert best_beam is not None
            jets.append(active.pop(best_beam))

    jets.sort(key=lambda jet: jet.pt(), reverse=True)
    return jets


def find_hard_higgs_labels(event: EventRecord) -> Dict[int, str]:
    """Map hard-process Higgs barcodes to labels H1/H2/H3."""
    higgs = [event.particles[b] for b in event.particle_order if event.particles[b].pdg_id == 25]
    primary = [p for p in higgs if p.status == 22]
    chosen = primary if len(primary) >= 3 else higgs

    labels: Dict[int, str] = {}
    for i, particle in enumerate(chosen[:3], start=1):
        labels[particle.barcode] = f"H{i}"
    return labels


def build_parent_lookup(event: EventRecord) -> Dict[int, List[int]]:
    parents: Dict[int, List[int]] = {}
    for barcode, particle in event.particles.items():
        if particle.prod_vertex is None:
            parents[barcode] = []
            continue
        incoming = event.incoming_by_vertex.get(particle.prod_vertex, [])
        parents[barcode] = [parent for parent in incoming if parent != barcode]
    return parents


def build_higgs_ancestry_resolver(event: EventRecord, hard_higgs: Dict[int, str]):
    """Return function barcode -> frozenset({'H1','H2','H3'}) of ancestor labels."""
    parents = build_parent_lookup(event)
    memo: Dict[int, FrozenSet[str]] = {}
    visiting: set[int] = set()

    def resolve(barcode: int) -> FrozenSet[str]:
        if barcode in memo:
            return memo[barcode]
        if barcode in hard_higgs:
            out = frozenset([hard_higgs[barcode]])
            memo[barcode] = out
            return out
        if barcode in visiting:
            return frozenset()

        visiting.add(barcode)
        labels: set[str] = set()
        for parent in parents.get(barcode, []):
            labels.update(resolve(parent))
        visiting.remove(barcode)

        out = frozenset(labels)
        memo[barcode] = out
        return out

    return resolve


def jet_higgs_percentages(jet: ClusterNode, label_lookup) -> Tuple[float, float, float]:
    """Return (H1%, H2%, H3%) for a clustered jet."""
    total = len(jet.constituents)
    if total == 0:
        return 0.0, 0.0, 0.0

    counts = {"H1": 0.0, "H2": 0.0, "H3": 0.0}
    for barcode in jet.constituents:
        labels = label_lookup(barcode)
        if not labels:
            continue
        weight = 1.0 / float(len(labels))
        for label in labels:
            if label in counts:
                counts[label] += weight

    return (
        100.0 * counts["H1"] / float(total),
        100.0 * counts["H2"] / float(total),
        100.0 * counts["H3"] / float(total),
    )


def jet_num_constituents(jet: ClusterNode) -> int:
    return len(jet.constituents)


def hadron_has_flavor(pdg_id: int, flavor: int) -> bool:
    """Return True if PDG ID corresponds to a hadron carrying 'flavor' quark."""
    a = abs(pdg_id)
    if a < 100 or a >= 1000000000:
        return False
    nq1 = (a // 10) % 10
    nq2 = (a // 100) % 10
    nq3 = (a // 1000) % 10
    return flavor in (nq1, nq2, nq3)


def build_flavor_resolver(event: EventRecord):
    """Return function barcode -> {'b','c','light'} from ancestry."""
    parents = build_parent_lookup(event)
    memo: Dict[int, str] = {}
    visiting: set[int] = set()

    def resolve(barcode: int) -> str:
        if barcode in memo:
            return memo[barcode]
        if barcode in visiting:
            return "light"

        visiting.add(barcode)
        particle = event.particles[barcode]
        apdg = abs(particle.pdg_id)

        if apdg == 5 or hadron_has_flavor(apdg, 5):
            flavor = "b"
        elif apdg == 4 or hadron_has_flavor(apdg, 4):
            flavor = "c"
        else:
            flavor = "light"
            for parent in parents.get(barcode, []):
                pf = resolve(parent)
                if pf == "b":
                    flavor = "b"
                    break
                if pf == "c":
                    flavor = "c"

        visiting.remove(barcode)
        memo[barcode] = flavor
        return flavor

    return resolve


def jet_truth_flavor(jet: ClusterNode, flavor_lookup) -> str:
    has_c = False
    for barcode in jet.constituents:
        f = flavor_lookup(barcode)
        if f == "b":
            return "b"
        if f == "c":
            has_c = True
    return "c" if has_c else "light"


def make_btag_flags(
    event_id: int,
    jet_index: int,
    truth_flavor: str,
    seed: int,
    effs: Dict[str, Dict[str, float]],
) -> Tuple[int, int, int]:
    """Return (loose, medium, tight) b-tag decisions."""
    local_seed = seed ^ ((event_id + 1) * 1000003 + jet_index * 9176)
    rng = random.Random(local_seed)
    u = rng.random()
    loose = 1 if u < effs["loose"][truth_flavor] else 0
    medium = 1 if u < effs["medium"][truth_flavor] else 0
    tight = 1 if u < effs["tight"][truth_flavor] else 0
    return loose, medium, tight


def analyze_event(
    event: EventRecord,
    radius: float,
    pt_min: float,
    use_truth_btag: bool,
    btag_seed: int,
    btag_effs: Dict[str, Dict[str, float]],
) -> EventResult:
    """Run jet clustering + Higgs ancestry fractions for a single event."""
    hard_higgs = find_hard_higgs_labels(event)
    ancestry = build_higgs_ancestry_resolver(event, hard_higgs)
    flavor_lookup = build_flavor_resolver(event) if use_truth_btag else None

    particle_nodes = [
        ClusterNode(vec=event.particles[b].vec, constituents=[b])
        for b in event.stable_barcodes
    ]

    jets = anti_kt_cluster(particle_nodes, radius)
    jets = [jet for jet in jets if jet.pt() >= pt_min]
    jet_constituent_counts = [jet_num_constituents(jet) for jet in jets]
    h_fracs = [jet_higgs_percentages(jet, ancestry) for jet in jets]
    truth_flavors = [jet_truth_flavor(jet, flavor_lookup) for jet in jets] if use_truth_btag else []
    btags = [
        make_btag_flags(event.event_id, i, truth_flavors[i - 1], btag_seed, btag_effs)
        for i in range(1, len(jets) + 1)
    ] if use_truth_btag else []

    return EventResult(
        event_id=event.event_id,
        hepmc_event_id=event.hepmc_event_id,
        n_constituents=len(event.stable_barcodes),
        jets=jets,
        jet_constituent_counts=jet_constituent_counts,
        h_fracs=h_fracs,
        truth_flavors=truth_flavors,
        btags=btags,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster anti-kt jets from a HepMC/HepMC.gz file.",
    )
    parser.add_argument("input", help="Input HepMC file (.hepmc or .hepmc.gz)")
    parser.add_argument(
        "--R",
        type=float,
        default=0.4,
        help="anti-kt radius parameter (default: 0.4)",
    )
    parser.add_argument(
        "--pt-min",
        type=float,
        default=0.0,
        help="Keep jets with pT >= this value (GeV, default: 0.0)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after this many events",
    )
    parser.add_argument(
        "--include-neutrinos",
        action="store_true",
        help="Include stable neutrinos in jet constituents (default: excluded)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output file for jet list",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress display",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes for event-level parallelization",
    )
    parser.add_argument(
        "--truth-btag",
        action="store_true",
        help="Enable generator-level truth b-tagging emulation columns",
    )
    parser.add_argument(
        "--btag-seed",
        type=int,
        default=12345,
        help="Random seed for truth b-tagging emulation",
    )
    parser.add_argument(
        "--btag-eff-b-loose",
        type=float,
        default=0.85,
        help="Truth-b tagging efficiency for b-jets at loose WP",
    )
    parser.add_argument(
        "--btag-eff-b-medium",
        type=float,
        default=0.70,
        help="Truth-b tagging efficiency for b-jets at medium WP",
    )
    parser.add_argument(
        "--btag-eff-b-tight",
        type=float,
        default=0.55,
        help="Truth-b tagging efficiency for b-jets at tight WP",
    )
    parser.add_argument(
        "--btag-eff-c-loose",
        type=float,
        default=0.40,
        help="Truth-b mistag efficiency for c-jets at loose WP",
    )
    parser.add_argument(
        "--btag-eff-c-medium",
        type=float,
        default=0.20,
        help="Truth-b mistag efficiency for c-jets at medium WP",
    )
    parser.add_argument(
        "--btag-eff-c-tight",
        type=float,
        default=0.10,
        help="Truth-b mistag efficiency for c-jets at tight WP",
    )
    parser.add_argument(
        "--btag-eff-light-loose",
        type=float,
        default=0.10,
        help="Truth-b mistag efficiency for light-jets at loose WP",
    )
    parser.add_argument(
        "--btag-eff-light-medium",
        type=float,
        default=0.01,
        help="Truth-b mistag efficiency for light-jets at medium WP",
    )
    parser.add_argument(
        "--btag-eff-light-tight",
        type=float,
        default=0.001,
        help="Truth-b mistag efficiency for light-jets at tight WP",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    btag_effs = {
        "loose": {"b": args.btag_eff_b_loose, "c": args.btag_eff_c_loose, "light": args.btag_eff_light_loose},
        "medium": {"b": args.btag_eff_b_medium, "c": args.btag_eff_c_medium, "light": args.btag_eff_light_medium},
        "tight": {"b": args.btag_eff_b_tight, "c": args.btag_eff_c_tight, "light": args.btag_eff_light_tight},
    }
    for wp in ("loose", "medium", "tight"):
        for flavor in ("b", "c", "light"):
            v = btag_effs[wp][flavor]
            if v < 0.0 or v > 1.0:
                raise ValueError(f"b-tag efficiency {wp}/{flavor} must be in [0,1], got {v}")
    for flavor in ("b", "c", "light"):
        if not (btag_effs["loose"][flavor] >= btag_effs["medium"][flavor] >= btag_effs["tight"][flavor]):
            raise ValueError(
                f"Expected loose >= medium >= tight for flavor '{flavor}'. "
                f"Got {btag_effs['loose'][flavor]}, {btag_effs['medium'][flavor]}, {btag_effs['tight'][flavor]}"
            )

    event_count = 0
    total_jets = 0

    out_fh = None
    writer = None
    if args.output:
        out_fh = open(args.output, "w", newline="", encoding="utf-8")
        writer = csv.writer(out_fh)
        writer.writerow(
            [
                "event",
                "hepmc_event",
                "jet",
                "pt",
                "eta",
                "y",
                "phi",
                "mass",
                "px",
                "py",
                "pz",
                "E",
                "num_constituents",
                "H1_percentage",
                "H2_percentage",
                "H3_percentage",
                "truth_flavor",
                "btag_loose",
                "btag_medium",
                "btag_tight",
            ]
        )

    show_progress = (not args.no_progress) and (tqdm is not None)
    if (not args.no_progress) and (tqdm is None):
        print("Note: tqdm is not installed; running without progress bar.")
    pbar = tqdm(
        total=args.max_events,
        desc="Clustering",
        unit="evt",
        dynamic_ncols=True,
    ) if show_progress else None

    def consume_result(result: EventResult) -> None:
        nonlocal event_count, total_jets
        event_count += 1
        total_jets += len(result.jets)
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(event_jets=len(result.jets), total_jets=total_jets, refresh=False)

        if writer is None:
            print(
                f"Event {result.event_id} (HepMC {result.hepmc_event_id}): "
                f"n_constituents={result.n_constituents} "
                f"n_jets={len(result.jets)}"
            )
            for i, (jet, (h1p, h2p, h3p)) in enumerate(zip(result.jets, result.h_fracs), start=1):
                flavor_txt = ""
                btag_txt = ""
                constituent_txt = f" nconst={result.jet_constituent_counts[i - 1]:4d}"
                if args.truth_btag:
                    flavor = result.truth_flavors[i - 1]
                    btl, btm, btt = result.btags[i - 1]
                    flavor_txt = f" flav={flavor:>5s}"
                    btag_txt = f" btag(L/M/T)=({btl}/{btm}/{btt})"
                print(
                    f"  jet {i:2d}  pt={jet.pt():10.4f}  eta={jet.eta():8.4f}  "
                    f"phi={jet.phi():8.4f}  m={jet.mass():9.4f}  "
                    f"H1={h1p:6.2f}% H2={h2p:6.2f}% H3={h3p:6.2f}%"
                    f"{constituent_txt}"
                    f"{flavor_txt}{btag_txt}"
                )
            return

        for i, (jet, (h1p, h2p, h3p)) in enumerate(zip(result.jets, result.h_fracs), start=1):
            truth_flavor = result.truth_flavors[i - 1] if args.truth_btag else ""
            if args.truth_btag:
                btag_loose, btag_medium, btag_tight = result.btags[i - 1]
            else:
                btag_loose, btag_medium, btag_tight = "", "", ""
            writer.writerow(
                [
                    result.event_id,
                    result.hepmc_event_id,
                    i,
                    f"{jet.pt():.10g}",
                    f"{jet.eta():.10g}",
                    f"{jet.rapidity():.10g}",
                    f"{jet.phi():.10g}",
                    f"{jet.mass():.10g}",
                    f"{jet.px:.10g}",
                    f"{jet.py:.10g}",
                    f"{jet.pz:.10g}",
                    f"{jet.e:.10g}",
                    result.jet_constituent_counts[i - 1],
                    f"{h1p:.6f}",
                    f"{h2p:.6f}",
                    f"{h3p:.6f}",
                    truth_flavor,
                    btag_loose,
                    btag_medium,
                    btag_tight,
                ]
            )

    try:
        if args.workers == 1:
            for i, event in enumerate(
                iter_hepmc2_events(args.input, include_neutrinos=args.include_neutrinos)
            ):
                if args.max_events is not None and i >= args.max_events:
                    break
                consume_result(analyze_event(event, args.R, args.pt_min, args.truth_btag, args.btag_seed, btag_effs))
        else:
            max_in_flight = max(2, args.workers * 4)
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                pending = {}
                next_to_consume = 0
                submitted = 0

                for event in iter_hepmc2_events(args.input, include_neutrinos=args.include_neutrinos):
                    if args.max_events is not None and submitted >= args.max_events:
                        break

                    pending[submitted] = executor.submit(
                        analyze_event,
                        event,
                        args.R,
                        args.pt_min,
                        args.truth_btag,
                        args.btag_seed,
                        btag_effs,
                    )
                    submitted += 1

                    while len(pending) >= max_in_flight and next_to_consume in pending:
                        consume_result(pending.pop(next_to_consume).result())
                        next_to_consume += 1

                while next_to_consume < submitted:
                    consume_result(pending.pop(next_to_consume).result())
                    next_to_consume += 1
    finally:
        if pbar is not None:
            pbar.close()
        if out_fh is not None:
            out_fh.close()

    print(
        f"Done. Processed {event_count} events, found {total_jets} jets "
        f"(R={args.R}, pt_min={args.pt_min})."
    )
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
