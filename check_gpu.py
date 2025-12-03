import torch

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device properties: {torch.cuda.get_device_properties(0)}')
else:
    print('No CUDA devices available')
    print('PyTorch was likely installed without CUDA support')