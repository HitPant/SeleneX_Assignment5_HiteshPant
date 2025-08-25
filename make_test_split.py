import random
from pathlib import Path

random.seed(42)  # reproducibility

val_file = Path("splits/val_cls.txt")
test_file = Path("splits/test_cls.txt")

lines = val_file.read_text().strip().splitlines()
n_test = max(1, int(0.5 * len(lines)))  # ~50% of val → test (adjust if needed)

test_lines = random.sample(lines, n_test)
remaining_val = [l for l in lines if l not in test_lines]

# write new files
test_file.write_text("\n".join(test_lines))
val_file.write_text("\n".join(remaining_val))

print(f"Created {test_file} with {len(test_lines)} rows")
print(f"Updated {val_file} with {len(remaining_val)} rows")
