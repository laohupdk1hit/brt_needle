import torch


print("CUDA Available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)

import transformers

print("Transformers version:", transformers.__version__)