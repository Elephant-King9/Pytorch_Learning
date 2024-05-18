import torch
# 测试MacOS版本支持
print(torch.backends.mps.is_available())
# 测试mps是否可用
print(torch.backends.mps.is_built())
