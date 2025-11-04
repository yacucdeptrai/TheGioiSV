import sys
from pathlib import Path


def main():
    print('=== PyTorch CUDA ===')
    try:
        import torch
        print(f'PyTorch version: {torch.__version__}')
        print(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'CUDA device count: {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                print(f'  [{i}] {torch.cuda.get_device_name(i)}')
            print(f'Current device: {torch.cuda.current_device()}')
        else:
            print('No CUDA device detected by PyTorch.')
    except Exception as e:
        print(f'PyTorch check failed: {e}')

    print('\n=== ONNX Runtime Providers ===')
    try:
        import onnxruntime as ort
        print(f'onnxruntime version: {ort.__version__}')
        print(f'Available providers: {ort.get_available_providers()}')
    except Exception as e:
        print(f'ONNX Runtime check failed: {e}')

    print('\n=== Ultralytics Device Resolution ===')
    try:
        import torch
        dev = None
        if torch.cuda.is_available():
            dev = 0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            dev = 'mps'
        else:
            dev = 'cpu'
        print(f'Ultralytics would use device: {dev}')
    except Exception as e:
        print(f'Ultralytics device check failed: {e}')


if __name__ == '__main__':
    main()
