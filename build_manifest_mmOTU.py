from pathlib import Path
import csv

# Paths
IM_DIR = Path("data/images")
SPLIT_DIR = Path("splits")
SPLITS = ["train", "val", "test"]

# Class mapping from OTU_2D order
id2name = {
    0: "chocolate_cyst",
    1: "serous_cystadenoma",
    2: "teratoma",
    3: "theca_cell_tumor",
    4: "simple_cyst",
    5: "normal_ovary",
    6: "mucinous_cystadenoma",
    7: "high_grade_serous",  # malignant
}
MALIGNANT_IDS = {7}

def read_cls(txt_path):
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # robust parse: first token is filename, last token is class id
            parts = line.replace(".JPG", "").replace(".PNG", "") \
                        .replace(".jpeg", "").replace(".JPEG", "") \
                        .replace(".jpg", "").replace(".png", "").split()
            img_stem = parts[0]
            cls_id = int(parts[-1])

            img_file = None
            for ext in [".JPG", ".PNG", ".jpg", ".png", ".jpeg", ".JPEG"]:
                p = IM_DIR / f"{img_stem}{ext}"
                if p.exists():
                    img_file = p.as_posix()
                    break
            if img_file is None:
                # fallback to .JPG path even if not present
                img_file = (IM_DIR / f"{img_stem}.JPG").as_posix()

            label_bin = 1 if cls_id in MALIGNANT_IDS else 0
            rows.append((img_file, label_bin, cls_id, id2name.get(cls_id, f"class_{cls_id}")))
    return rows

def write_manifest(rows, out_csv, with_names=False):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        if with_names:
            w.writerow(["image_path", "label", "orig_cls_id", "orig_cls_name"])
            for r in rows:
                w.writerow(r)
        else:
            w.writerow(["image_path", "label"])
            for img_path, label_bin, _, _ in rows:
                w.writerow([img_path, label_bin])

def main():
    all_rows = []
    SPLIT_DIR = Path("splits")
    for split in ["train", "val", "test"]:
        p = SPLIT_DIR / f"{split}_cls.txt"
        if p.exists():
            print(f"Reading {p} ...")
            all_rows += read_cls(p)
        else:
            print(f"Missing {p}, skipping.")

    # write combined manifests
    with open("data/manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label"])
        for img_path, label_bin, _, _ in all_rows:
            w.writerow([img_path, label_bin])

    with open("data/manifest_with_names.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "orig_cls_id", "orig_cls_name"])
        for row in all_rows:
            w.writerow(row)

    print(f"Wrote manifests. Combined rows: {len(all_rows)}")

if __name__ == "__main__":
    main()
