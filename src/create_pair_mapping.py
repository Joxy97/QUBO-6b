def create_pair_mapping(num_jets: int):    
    k = 0
    mapping = {}
    for i in range(num_jets):
        for j in range(i+1, num_jets):
            mapping[f'{k}'] = (i, j)
            k += 1

    return mapping