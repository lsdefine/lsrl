import torch, transformers
assert torch.__version__ >= "2.0.0", f"Need PyTorch 2.0+, got {torch.__version__}"
assert transformers.__version__ >= "4.20.0", f"Need transformers 4.20+, got {transformers.__version__}"

from .lsrl import LSRL, LSTrainer, LSCPUTrainer, DeepSpeedTrainer, GenLogRecorder
from .cpuadamw import CPUAdamW, DistributedCPUAdamW
from .ref_server import RefServer
from .patch_for_multi_gpus import patch_qwen2_for_multi_gpus
from .utils import save_model, json_to_bytes_list, bytes_list_to_json

__version__ = "0.1.0"
__all__ = ["LSRL", "LSTrainer", "LSCPUTrainer", "DeepSpeedTrainer", 
           "GenLogRecorder", "CPUAdamW", "DistributedCPUAdamW", "RefServer"]  

__all__ += ["patch_qwen2_for_multi_gpus", "save_model", "json_to_bytes_list", "bytes_list_to_json"]
