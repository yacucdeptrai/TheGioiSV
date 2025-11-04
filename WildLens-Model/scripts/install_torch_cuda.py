import argparse
import subprocess
import sys
import shutil
import re


def run(cmd: list[str], check: bool = True) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd) if not check else subprocess.check_call(cmd)


def detect_cuda_from_nvidia_smi() -> str | None:
    """Returns CUDA major.minor string from nvidia-smi, e.g., '12.1' or '12.4'. None if unavailable."""
    exe = shutil.which('nvidia-smi')
    if not exe:
        return None
    try:
        out = subprocess.check_output([exe], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return m.group(1) if m else None


def choose_pytorch_index(cuda_ver: str | None) -> str:
    """Map detected CUDA to appropriate PyTorch wheel index URL. Defaults to cu121 if unknown."""
    # Known indices as of PyTorch 2.4+; adjust as PyTorch evolves
    if cuda_ver:
        try:
            major, minor = (int(x) for x in cuda_ver.split('.')[:2])
            if (major, minor) >= (12, 4):
                return 'https://download.pytorch.org/whl/cu124'
            # Fallback for 12.1–12.3
            return 'https://download.pytorch.org/whl/cu121'
        except Exception:
            pass
    return 'https://download.pytorch.org/whl/cu121'


def main():
    parser = argparse.ArgumentParser(description='Install CUDA-enabled PyTorch matching your NVIDIA driver')
    parser.add_argument('--yes', action='store_true', help='Run installation commands instead of printing them')
    parser.add_argument('--index-url', type=str, default='', help='Override PyTorch wheel index URL (e.g., cu121 or cu124)')
    parser.add_argument('--extra', type=str, default='', help='Extra pip args, e.g., --extra-index-url ...')
    args = parser.parse_args()

    print('=== Detecting NVIDIA/CUDA ===')
    cuda = detect_cuda_from_nvidia_smi()
    if cuda:
        print(f'Detected CUDA via nvidia-smi: {cuda}')
    else:
        print('Could not detect CUDA via nvidia-smi. You can still install a CUDA wheel if your driver is compatible.')

    index = args.index_url or choose_pytorch_index(cuda)
    print(f'Chosen PyTorch index: {index}')

    cmds = [
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'],
        [sys.executable, '-m', 'pip', 'install', '--index-url', index, 'torch', 'torchvision', 'torchaudio'] + (args.extra.split() if args.extra else []),
    ]

    print('\n=== Planned commands ===')
    for c in cmds:
        print(' ', ' '.join(c))

    if args.yes:
        print('\nExecuting...')
        for c in cmds:
            code = run(c, check=False)
            if code != 0:
                print(f'Command failed with exit code {code}: {" ".join(c)}')
                sys.exit(code)
        print('Done.')
    else:
        print('\nDry run only. Re-run with --yes to execute the commands.')

    print('\nPost-install verification (run manually):')
    print('  python - <<"PY"')
    print('  import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())')
    print('  PY')


if __name__ == '__main__':
    main()
