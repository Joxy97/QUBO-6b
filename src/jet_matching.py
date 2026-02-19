import numpy as np

# ----------------------------
# Geometry helpers
# ----------------------------
def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return np.arctan2(np.sin(dphi), np.cos(dphi))

def delta_r_matrix(b_eta, b_phi, j_eta, j_phi):
    # b_eta/b_phi: (Nb,), j_eta/j_phi: (Nj,)
    b_eta = b_eta[:, None]
    b_phi = b_phi[:, None]
    j_eta = j_eta[None, :]
    j_phi = j_phi[None, :]
    deta = b_eta - j_eta
    dphi = delta_phi(b_phi, j_phi)
    return np.sqrt(deta**2 + dphi**2)  # (Nb, Nj)

# ----------------------------
# Truth ancestry helpers
# ----------------------------
def climb_to_first_ancestor_with_pdg(idx, mother_idx, pdgId, target_pdg=25, max_steps=50):
    """
    Starting from GenPart index idx, climb mother links until you hit pdg==target_pdg.
    Returns the GenPart index of that ancestor, or -1 if not found.
    """
    cur = int(idx)
    for _ in range(max_steps):
        mom = int(mother_idx[cur])
        if mom < 0:
            return -1
        if int(pdgId[mom]) == int(target_pdg):
            return mom
        cur = mom
    return -1

# ----------------------------
# Pair mapping (given)
# ----------------------------
def create_pair_mapping(num_jets: int):
    k = 0
    mapping = {}
    for i in range(num_jets):
        for j in range(i + 1, num_jets):
            mapping[f"{k}"] = (i, j)
            k += 1
    return mapping

# ----------------------------
# Assignment core
# ----------------------------
def map_jets_to_b_and_pairs_for_event(
    GenPart_pdgId_evt,
    GenPart_status_evt,
    GenPart_genPartIdxMother_evt,
    GenPart_eta_evt,
    GenPart_phi_evt,
    GenJet_eta_evt,
    GenJet_phi_evt,
    num_higgs_expected=3,
):
    """
    No dR cut-off:
      - Find b-quarks with status==23 (and pdgId=±5).
      - For each b, find its Higgs (pdgId=25) ancestor via mother chain.
      - Label Higgses as 0,1,2 (stable ordering by Higgs GenPart index).
      - Match each b to a distinct jet by global minimum total ΔR (Hungarian).
      - Jet label = higgs_id (0/1/2) for matched jets, else -1.
      - Pair_truthAssignment = 1 for the unique jet-pair per Higgs (two jets matched to that Higgs), else 0.
    """

    # ---- 1) select status-23 b quarks
    pdg = np.asarray(GenPart_pdgId_evt)
    st  = np.asarray(GenPart_status_evt)
    mom = np.asarray(GenPart_genPartIdxMother_evt)

    b_idx = np.where((st == 23) & (np.abs(pdg) == 5))[0]
    if b_idx.size == 0:
        # no b quarks found
        nj = len(GenJet_eta_evt)
        jet_labels = -1 * np.ones(nj, dtype=np.int8)
        pair_map = create_pair_mapping(nj)
        pair_truth = np.zeros(len(pair_map), dtype=np.int8)
        return jet_labels, pair_truth

    # ---- 2) for each b, find which Higgs it came from
    higgs_anc = np.array(
        [climb_to_first_ancestor_with_pdg(i, mom, pdg, target_pdg=25) for i in b_idx],
        dtype=np.int32
    )

    # Keep only b's that actually trace back to a Higgs=25
    good = higgs_anc >= 0
    b_idx = b_idx[good]
    higgs_anc = higgs_anc[good]

    if b_idx.size == 0:
        nj = len(GenJet_eta_evt)
        jet_labels = -1 * np.ones(nj, dtype=np.int8)
        pair_map = create_pair_mapping(nj)
        pair_truth = np.zeros(len(pair_map), dtype=np.int8)
        return jet_labels, pair_truth

    # ---- 3) define Higgs labels 0/1/2 (stable ordering by Higgs GenPart index)
    uniq_h = np.unique(higgs_anc)
    uniq_h_sorted = np.sort(uniq_h)  # stable + reproducible
    # If you *expect* 3 Higgs, you can assert:
    # if len(uniq_h_sorted) != num_higgs_expected: print("warning ...")
    higgs_to_label = {int(h): int(k) for k, h in enumerate(uniq_h_sorted[:num_higgs_expected])}

    b_higgs_label = np.array([higgs_to_label.get(int(h), -1) for h in higgs_anc], dtype=np.int8)

    # Optional: keep only b's assigned to labels 0..2 (if extra Higgs-like ancestors appear)
    keep = b_higgs_label >= 0
    b_idx = b_idx[keep]
    b_higgs_label = b_higgs_label[keep]

    # ---- 4) build ΔR matrix between these b's and all jets
    b_eta = np.asarray(GenPart_eta_evt)[b_idx].astype(np.float32)
    b_phi = np.asarray(GenPart_phi_evt)[b_idx].astype(np.float32)

    j_eta = np.asarray(GenJet_eta_evt).astype(np.float32)
    j_phi = np.asarray(GenJet_phi_evt).astype(np.float32)

    nb = b_eta.shape[0]
    nj = j_eta.shape[0]

    # If fewer jets than b's, we still do "best we can" by padding jets (rare if you have >=6 jets)
    # But your stated case is nj >= nb.
    dR = delta_r_matrix(b_eta, b_phi, j_eta, j_phi)  # (nb, nj)

    # ---- 5) match b -> jet with minimum total ΔR (no cut-off)
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(dR)  # assigns min(nb, nj) pairs
        # If nj >= nb: we get nb unique jets, one per b.
    except Exception:
        # Fallback: greedy unique matching (works without scipy)
        row_ind = []
        col_ind = []
        used_j = set()
        for i in range(nb):
            # pick closest unused jet
            order = np.argsort(dR[i])
            picked = None
            for j in order:
                if int(j) not in used_j:
                    picked = int(j)
                    break
            if picked is None:
                break
            used_j.add(picked)
            row_ind.append(i)
            col_ind.append(picked)
        row_ind = np.array(row_ind, dtype=int)
        col_ind = np.array(col_ind, dtype=int)

    # ---- 6) build jet truth labels: 0/1/2 or -1
    jet_labels = -1 * np.ones(nj, dtype=np.int8)
    # row_ind indexes b's (0..nb-1); col_ind are jet indices
    for bi, ji in zip(row_ind, col_ind):
        jet_labels[int(ji)] = int(b_higgs_label[int(bi)])

    # ---- 7) build pair truth assignment using your mapping
    pair_map = create_pair_mapping(nj)
    pair_truth = np.zeros(len(pair_map), dtype=np.int8)

    # For each Higgs label, find the two jets matched to it, then mark that pair = 1
    for h in range(num_higgs_expected):
        jets_h = np.where(jet_labels == h)[0]
        if jets_h.size != 2:
            continue  # unresolved or extra weirdness
        a, b = int(jets_h[0]), int(jets_h[1])
        i, j = (a, b) if a < b else (b, a)

        # find the pair index k such that mapping[k] == (i,j)
        # (fast way: compute k analytically, but simplest is lookup by scan)
        for k_str, (p, q) in pair_map.items():
            if p == i and q == j:
                pair_truth[int(k_str)] = 1
                break

    return jet_labels, pair_truth


# ----------------------------
# Example: run over all events (jagged arrays dtype=object)
# ----------------------------
def build_truth_assignments_all_events(
    GenPart_pdgId, GenPart_status, GenPart_genPartIdxMother, GenPart_eta, GenPart_phi,
    GenJet_eta, GenJet_phi
):
    n_events = len(GenJet_eta)
    GenJet_truthAssignment = np.empty(n_events, dtype=object)
    Pair_truthAssignment   = np.empty(n_events, dtype=object)

    for ev in range(n_events):
        jet_lab, pair_lab = map_jets_to_b_and_pairs_for_event(
            GenPart_pdgId[ev],
            GenPart_status[ev],
            GenPart_genPartIdxMother[ev],
            GenPart_eta[ev],
            GenPart_phi[ev],
            GenJet_eta[ev],
            GenJet_phi[ev],
            num_higgs_expected=3,
        )
        GenJet_truthAssignment[ev] = jet_lab
        Pair_truthAssignment[ev]   = pair_lab

    return GenJet_truthAssignment, Pair_truthAssignment


# Usage:
'''
GenJet_truthAssignment, Pair_truthAssignment = build_truth_assignments_all_events(
    GenPart_pdgId, GenPart_status, GenPart_genPartIdxMother, GenPart_eta, GenPart_phi,
    GenJet_eta, GenJet_phi
    )

# Then for an event:
ev = 0
print(GenJet_truthAssignment[ev])  # array of size nJets with values {0,1,2,-1}
print(Pair_truthAssignment[ev])    # array of size nPairs with 0/1

'''