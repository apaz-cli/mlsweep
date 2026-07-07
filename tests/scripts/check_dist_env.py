"""Check distributed environment variables (tolerant mode).

Reports environment variables without asserting — always exits 0 so that
test sweeps don't fail when dist env vars are not set.
"""
import os
import sys

for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
    print(f"{var}={os.environ.get(var, '(unset)')}")

gpu_rank = os.environ.get("MLSWEEP_GPU_RANK", "0")
print(f"MLSWEEP_GPU_RANK={gpu_rank}")

# Log a metric via MLSweepLogger (no-op when MLSWEEP_WORKER_SOCKET is unset)
from mlsweep.logger import MLSweepLogger
logger = MLSweepLogger()
logger.log({"dist_check": float(gpu_rank or "0")})
logger.close()

print(f"Rank {gpu_rank} finished.")
sys.exit(0)
