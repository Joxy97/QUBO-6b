import uproot
import awkward as ak
import numpy as np


def root_to_numpy(path: str, tree_name="Events", num_events: int | None = None):
    with uproot.open(path) as f:
        tree = f[tree_name]
        arrays = tree.arrays(entry_start=0, entry_stop=num_events, library="ak")

    out = {}

    for name in arrays.fields:
        arr = arrays[name]

        # Try a direct conversion first (works for flat arrays)
        try:
            flat = ak.to_numpy(arr)

            # Jagged often turns into dtype=object (or raises before this)
            if flat.dtype != object:
                out[name] = flat
                continue
        except Exception:
            pass

        # Fallback: treat as jagged and make numpy array-of-numpy-arrays
        out[name] = np.array([np.asarray(x) for x in arr], dtype=object)

    return out



# Usage example
'''
data = root_to_numpy("data/HHH.root", num_events=1000)

GenPart_mass = data["GenPart_mass"]

print(type(GenPart_mass))           # numpy.ndarray
print(type(GenPart_mass[0]))        # numpy.ndarray
print(GenPart_mass[0][[1,2,5]])     # Works!
'''