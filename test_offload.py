import torch
from sentence_transformers import SentenceTransformer
import time

def print_mem(stage):
    allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[{stage}] CUDA VRAM Allocated: {allocated:.2f} GB")

print_mem("Start")
model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device="cuda", model_kwargs={"torch_dtype": torch.float16})
print_mem("Loaded directly to CUDA")
time.sleep(1)

# Offload to CPU
print("Offloading to CPU...")
model.to("cpu")
torch.cuda.empty_cache()
print_mem("After offload & cache empty")
time.sleep(1)

# Back to CUDA
print("Moving back to CUDA...")
model.to("cuda")
print_mem("Back on CUDA")
