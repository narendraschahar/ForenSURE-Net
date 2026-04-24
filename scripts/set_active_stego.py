import shutil
import sys
from pathlib import Path

BASE = Path("data/BOSSBase")
ACTIVE = BASE / "stego"

VALID = [
    "stego_lsb",
    "stego_wow_04",
    "stego_suniward_04"
]

if len(sys.argv) != 2:
    print("Usage: python scripts/set_active_stego.py stego_lsb")
    sys.exit(1)

source_name = sys.argv[1]

if source_name not in VALID:
    print("Invalid option.")
    print("Choose from:", VALID)
    sys.exit(1)

source = BASE / source_name

if not source.exists():
    print(f"Folder not found: {source}")
    sys.exit(1)

ACTIVE.mkdir(parents=True, exist_ok=True)

# Clear current active folder
for f in ACTIVE.glob("*"):
    if f.is_file():
        f.unlink()

# Copy selected files
files = list(source.glob("*.pgm"))

for f in files:
    shutil.copy2(f, ACTIVE / f.name)

print(f"Activated: {source_name}")
print(f"Files copied: {len(files)}")