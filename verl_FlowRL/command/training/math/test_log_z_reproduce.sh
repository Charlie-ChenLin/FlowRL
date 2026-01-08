python - <<'PY'
import torch, hashlib
from verl.workers.fsdp_workers import ProjZModule

def hash_proj_z(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = ProjZModule(3584, num_layers=3)
    vec = torch.nn.utils.parameters_to_vector(m.parameters()).detach().cpu().numpy().tobytes()
    return hashlib.sha256(vec).hexdigest()

h1 = hash_proj_z(1)
h2 = hash_proj_z(1)
print("hash1:", h1)
print("hash2:", h2)
print("equal:", h1 == h2)
PY
# sha256=ec3a91cde475c45c3eae49505b48bc56466d9c0e445e3ad38f426d894e2e8d62