import numpy as np

from src.create_pair_mapping import create_pair_mapping


def _delta_phi(phi1: float, phi2: float) -> float:
    """Return wrapped delta-phi in [-pi, pi]."""
    return float(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def _pair_invariant_mass(
    pt1: float, eta1: float, phi1: float, mass1: float,
    pt2: float, eta2: float, phi2: float, mass2: float,
) -> float:
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(np.maximum(mass1**2 + px1**2 + py1**2 + pz1**2, 0.0))

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(np.maximum(mass2**2 + px2**2 + py2**2 + pz2**2, 0.0))

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    mass_sq = e**2 - px**2 - py**2 - pz**2
    return float(np.sqrt(np.maximum(mass_sq, 0.0)))


def _pair_delta_r(
    pt1: float, eta1: float, phi1: float, mass1: float,
    pt2: float, eta2: float, phi2: float, mass2: float,
) -> float:
    d_eta = eta1 - eta2
    d_phi = _delta_phi(phi1, phi2)
    return float(np.sqrt(d_eta**2 + d_phi**2))


def _pair_pt_balance(
    pt1: float, eta1: float, phi1: float, mass1: float,
    pt2: float, eta2: float, phi2: float, mass2: float,
) -> float:
    denom = pt1 + pt2
    if denom <= 0.0:
        return 0.0
    return float(np.abs(pt1 - pt2) / denom)


def _pair_pt(
    pt1: float, eta1: float, phi1: float, mass1: float,
    pt2: float, eta2: float, phi2: float, mass2: float,
) -> float:
    px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    return float(np.hypot(px, py))


def _pair_eta(
    pt1: float, eta1: float, phi1: float, mass1: float,
    pt2: float, eta2: float, phi2: float, mass2: float,
) -> float:
    px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    pz = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)

    pt = np.hypot(px, py)
    if pt <= 0.0:
        return 0.0
    return float(np.arcsinh(pz / pt))


def _build_pair_quantity(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass, quantity_fn):
    """Build one jagged per-event array of pair-level values."""
    out = []

    for event_pt, event_eta, event_phi, event_mass in zip(
        GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass
    ):
        pt = np.asarray(event_pt, dtype=np.float64)
        eta = np.asarray(event_eta, dtype=np.float64)
        phi = np.asarray(event_phi, dtype=np.float64)
        mass = np.asarray(event_mass, dtype=np.float64)

        num_jets = len(pt)
        pair_map = create_pair_mapping(num_jets)
        event_values = np.empty(len(pair_map), dtype=np.float64)

        for pair_idx, (i, j) in enumerate(pair_map.values()):
            event_values[pair_idx] = quantity_fn(
                pt[i], eta[i], phi[i], mass[i],
                pt[j], eta[j], phi[j], mass[j],
            )

        out.append(event_values)

    return np.array(out, dtype=object)


def build_pair_masses(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass):
    return _build_pair_quantity(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass, _pair_invariant_mass)


def build_pair_delta_r(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass):
    return _build_pair_quantity(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass, _pair_delta_r)


def build_pair_pt_balance(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass):
    return _build_pair_quantity(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass, _pair_pt_balance)


def build_pair_pt(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass):
    return _build_pair_quantity(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass, _pair_pt)


def build_pair_eta(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass):
    return _build_pair_quantity(GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass, _pair_eta)
