import argparse
from pathlib import Path
import yaml
from typing import Dict, List

# Global default class list as fallback
TARGET_CLASSES_30 = [
    'Dog','Cat','Horse','Cow','Sheep','Pig','Chicken','Duck','Bird','Elephant',
    'Lion','Tiger','Bear','Monkey','Deer','Fox','Wolf','Rabbit','Squirrel','Giraffe',
    'Zebra','Kangaroo','Panda','Koala','Raccoon','Penguin','Dolphin','Whale','Turtle','Frog'
]


def load_yaml(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_name_to_id(names: List[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def remap_label_file(file_path: Path, new_id: int, backup: bool = True) -> int:
    """Rewrite first integer token (class id) on each non-empty line to new_id.
    Returns number of lines rewritten.
    """
    text = file_path.read_text(encoding='utf-8')
    lines = text.splitlines()
    out_lines = []
    changed = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            out_lines.append(ln)
            continue
        parts = s.split()
        try:
            int(parts[0])  # validate
        except Exception:
            # keep line as-is if first token is not an int
            out_lines.append(ln)
            continue
        parts[0] = str(new_id)
        out_lines.append(' '.join(parts))
        changed += 1
    if backup:
        file_path.with_suffix(file_path.suffix + '.bak').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    file_path.write_text('\n'.join(out_lines) + ('\n' if out_lines else ''), encoding='utf-8')
    return changed


def detect_species_dir(path: Path, valid_species: Dict[str, int]) -> tuple[str, int] | None:
    """Return (species_name, class_id) inferred from directory names, or None if not matched.
    Looks up the nearest ancestor directory whose name equals a key in valid_species.
    """
    p = path
    for _ in range(6):  # up 6 levels is plenty: .../<Species>/<split>/labels/file.txt
        p = p.parent
        if p is None:
            break
        if p.name in valid_species:
            return p.name, valid_species[p.name]
    return None


def scan_unique_ids(root: Path) -> List[int]:
    ids = set()
    for f in root.rglob('labels/*.txt'):
        for line in f.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s:
                continue
            tok = s.split()[0]
            if tok.isdigit():
                ids.add(int(tok))
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(description='Remap YOLO label class ids based on species folder name')
    parser.add_argument('--data-yaml', type=str, default=str(Path(__file__).resolve().parent.parent / 'data' / 'data.yaml'),
                        help='Path to the centralized data.yaml to read class names from')
    parser.add_argument('--data-root', type=str, default=str(Path(__file__).resolve().parent.parent / 'data'),
                        help='Root folder containing <Species>/{train,valid,test}/labels')
    parser.add_argument('--dry-run', action='store_true', help='Do not modify files, just report what would change')
    parser.add_argument('--no-backup', action='store_true', help='Do not write .bak files next to labels')
    parser.add_argument('--verbose', action='store_true', help='Print every file touched')
    args = parser.parse_args()

    data_yaml_path = Path(args.data_yaml)
    data_root = Path(args.data_root)

    # Determine class names
    names = TARGET_CLASSES_30
    if data_yaml_path.exists():
        try:
            cfg = load_yaml(data_yaml_path)
            if isinstance(cfg.get('names'), list) and cfg.get('names'):
                names = cfg['names']
                print(f'Loaded {len(names)} class names from {data_yaml_path}')
            else:
                print('WARN: data.yaml has no names list; falling back to default 30-class list')
        except Exception as e:
            print(f'WARN: Failed to read data.yaml ({e}); using default 30-class list')
    else:
        print('WARN: data.yaml not found; using default 30-class list')

    name2id = build_name_to_id(names)

    # Quick pre-scan
    before_ids = scan_unique_ids(data_root)
    print(f'Unique class ids before remap: {before_ids}')

    total_files = 0
    total_lines = 0

    for lbl in data_root.rglob('labels/*.txt'):
        det = detect_species_dir(lbl, name2id)
        if not det:
            continue
        species, gid = det
        if args.dry_run:
            if args.verbose:
                print(f'[DRY] {lbl} -> {species} ({gid})')
            total_files += 1
            continue
        changed = remap_label_file(lbl, gid, backup=not args.no_backup)
        total_files += 1
        total_lines += changed
        if args.verbose:
            print(f'Remapped {lbl} to id {gid} ({species}) lines={changed}')

    # Post-scan
    after_ids = scan_unique_ids(data_root)
    print(f'Unique class ids after remap: {after_ids}')
    print(f'Files processed: {total_files}, label lines rewritten: {total_lines}')
    if args.dry_run:
        print('Dry run finished. Re-run without --dry-run to apply changes.')


if __name__ == '__main__':
    main()
