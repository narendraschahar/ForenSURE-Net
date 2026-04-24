import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--bossbase_input", type=str, required=True)
args = parser.parse_args()

source = Path(args.bossbase_input)
target = Path("data/BOSSBase/cover")
target.mkdir(parents=True, exist_ok=True)

files = sorted(source.glob("*.pgm"))

if len(files) == 0:
    raise ValueError(f"No .pgm files found at {source}")

for f in tqdm(files, desc="Copying BOSSBase covers"):
    shutil.copy2(f, target / f.name)

print("Copied cover images:", len(list(target.glob("*.pgm"))))