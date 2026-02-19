def create_pair_mapping(num_jets: int):    
    k = 0
    mapping = {}
    for i in range(num_jets):
        for j in range(i+1, num_jets):
            mapping[f'{k}'] = (i, j)
            k += 1

    return mapping

def pairs_that_contain_jet(jet_index, mapping):
    return [key for key, value in mapping.items() if jet_index in value]

def build_jet_to_pairs(mapping, num_jets):
    jet_to_pairs = {k: [] for k in range(num_jets)}
    for pair_idx, (i, j) in mapping.items():
        jet_to_pairs[i].append(pair_idx)
        jet_to_pairs[j].append(pair_idx)
    return jet_to_pairs

