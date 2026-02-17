from __future__ import annotations

import csv
import os
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import awkward as ak
import uproot


PDG_CSV_PATH = Path(__file__).with_name("pdg2025_table.csv")


def load_pdg_map(csv_path: Path) -> dict[int, str]:
    """Load PDG id -> particle name from pdg2025_table.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"PDG table not found: {csv_path}")

    pdg_map: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = (row.get("ID") or "").strip()
            if not raw_id:
                continue

            try:
                pdgid = int(raw_id)
            except ValueError:
                continue

            name = (row.get("Name") or "").strip()
            latex = (row.get("Latex") or "").strip()
            pdg_map[pdgid] = name or latex or str(pdgid)

    return pdg_map


PDG_MAP = load_pdg_map(PDG_CSV_PATH)


def pdg_name(pdgid: int) -> str:
    return PDG_MAP.get(int(pdgid), str(int(pdgid)))


def build_daughters(mother_idx):
    daughters = defaultdict(list)
    for i, m in enumerate(mother_idx):
        if m is None:
            continue
        m = int(m)
        if m >= 0:
            daughters[m].append(i)
    return daughters


def descendants_from_roots(roots, daughters):
    keep = set()
    dq = deque(roots)
    while dq:
        u = dq.popleft()
        if u in keep:
            continue
        keep.add(u)
        for v in daughters.get(u, []):
            if v not in keep:
                dq.append(v)
    return keep


def dot_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def get_float(values, index: int) -> float:
    if values is None:
        return float("nan")
    try:
        return float(values[index])
    except (TypeError, ValueError, IndexError):
        return float("nan")


def particle_label(i, pdgId, mother, status, pt, mass, newline: str, escape_name: bool) -> str:
    pid = int(pdgId[i])
    name = pdg_name(pid)
    if escape_name:
        name = dot_escape(name)

    st = int(status[i]) if status is not None else -999
    pti = get_float(pt, i)
    massi = get_float(mass, i)

    return (
        f"{i}  {name}{newline}"
        f"PDG {pid}  st {st}{newline}"
        f"pt {pti:.3g}  mass {massi:.3g}"
    )


def write_dot(out_dot, event, pdgId, mother, status, pt, mass, daughters, keep_set=None, pt_min=0.0):
    n = len(pdgId)

    def keep_node(i: int) -> bool:
        if keep_set is not None and i not in keep_set:
            return False
        pti = get_float(pt, i)
        if pt is not None and pti < pt_min:
            return False
        return True

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("digraph Decay {\n")
        f.write("  rankdir=TB;\n")
        f.write('  node [shape=box, fontsize=10];\n')
        f.write(f'  labelloc="t"; label="Event {event} GenPart decay";\n')

        for i in range(n):
            if not keep_node(i):
                continue
            label = particle_label(i, pdgId, mother, status, pt, mass, newline="\\n", escape_name=True)
            f.write(f'  n{i} [label="{label}"];\n')

        for m, ds in daughters.items():
            if not keep_node(m):
                continue
            for d in ds:
                if not keep_node(d):
                    continue
                f.write(f"  n{m} -> n{d};\n")

        f.write("}\n")


def render_png_with_graphviz(dot_path, png_path):
    dot_exe = shutil.which("dot")
    if not dot_exe:
        return False
    try:
        subprocess.run([dot_exe, "-Tpng", dot_path, "-o", png_path], check=True)
        return os.path.exists(png_path)
    except Exception:
        return False


def render_png_with_networkx(png_path, nodes, edges, labels):
    # Pure-Python fallback: networkx + matplotlib
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # hierarchical-ish layout (if graphviz layout isn't available)
    # Try "graphviz_layout" if pygraphviz/pydot installed; else spring_layout.
    try:
        from networkx.drawing.nx_pydot import graphviz_layout

        pos = graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=1)

    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, arrows=True, with_labels=False, node_size=800)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return os.path.exists(png_path)


def draw_decay_chain(
    file_path="HHH.root",
    tree_name="Events",
    event=0,
    out_prefix="decay",
    pt_min=0.0,
    only_from_pdg=25,  # 25=Higgs, 0=everything
    make_png=True,
):
    branches = ["GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_status", "GenPart_pt", "GenPart_mass"]

    with uproot.open(file_path) as f:
        t = f[tree_name]
        arr = t.arrays(branches, entry_start=event, entry_stop=event + 1, library="ak")

    pdgId = ak.to_list(arr["GenPart_pdgId"][0])
    mother = ak.to_list(arr["GenPart_genPartIdxMother"][0])
    status = ak.to_list(arr["GenPart_status"][0])
    pt = ak.to_list(arr["GenPart_pt"][0])
    mass = ak.to_list(arr["GenPart_mass"][0])

    daughters = build_daughters(mother)

    keep_set = None
    if only_from_pdg != 0:
        roots = [i for i, pid in enumerate(pdgId) if int(pid) == int(only_from_pdg)]
        keep_set = descendants_from_roots(roots, daughters)

    def keep_node(i: int) -> bool:
        if keep_set is not None and i not in keep_set:
            return False
        pti = get_float(pt, i)
        if pt is not None and pti < pt_min:
            return False
        return True

    out_dot = f"{out_prefix}_evt{event}.dot"
    out_png = f"{out_prefix}_evt{event}.png"

    # Always write DOT
    write_dot(out_dot, event, pdgId, mother, status, pt, mass, daughters, keep_set=keep_set, pt_min=pt_min)
    print(f"[ok] wrote DOT: {out_dot}")

    png_path = None
    if make_png:
        # 1) Try real Graphviz (best)
        ok = render_png_with_graphviz(out_dot, out_png)
        if ok:
            png_path = out_png
            print(f"[ok] wrote PNG via Graphviz: {out_png}")
        else:
            # 2) Pure Python fallback render
            nodes = [i for i in range(len(pdgId)) if keep_node(i)]
            edges = []
            for m, ds in daughters.items():
                if not keep_node(m):
                    continue
                for d in ds:
                    if keep_node(d):
                        edges.append((m, d))

            labels = {
                i: particle_label(i, pdgId, mother, status, pt, mass, newline="\n", escape_name=False)
                for i in nodes
            }

            try:
                ok2 = render_png_with_networkx(out_png, nodes, edges, labels)
                if ok2:
                    png_path = out_png
                    print(f"[ok] wrote PNG via networkx/matplotlib: {out_png}")
                else:
                    print("[warn] PNG render failed (fallback). DOT is still valid.")
            except ImportError:
                print("[warn] networkx/matplotlib not installed, cannot render PNG without Graphviz.")
                print("       Install: pip install networkx matplotlib pydot")
                print("       Or install Graphviz and ensure 'dot' is in PATH.")

    return out_dot, png_path


def print_tree(root, daughters, pdgId, depth=0, max_depth=12):
    if depth > max_depth:
        print("  " * depth + "...")
        return
    print("  " * depth + f"{root}: {pdg_name(int(pdgId[root]))} ({int(pdgId[root])})")
    for d in daughters.get(root, []):
        print_tree(d, daughters, pdgId, depth + 1, max_depth)


# Usage example
'''
draw_decay_chain(
    file_path="data/root_files/HHH.root",
    tree_name="Events",
    event=0,
    out_prefix="hhh_decay",
    pt_min=0.0,
    only_from_pdg=25,
    make_png=True,
)
'''
