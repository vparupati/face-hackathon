from sklearn.datasets import fetch_lfw_people
import numpy as np, os
from PIL import Image

def save_person(images, names, person, outdir):
    os.makedirs(outdir, exist_ok=True)
    idxs = np.where(names == person)[0]
    for i, idx in enumerate(idxs):
        img = Image.fromarray((images[idx] * 255).astype("uint8"))
        img = img.convert("RGB")
        img.save(os.path.join(outdir, f"{person.replace(' ', '_')}_{i:03d}.jpg"))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_per_person", type=int, default=20, help="Only include people with at least this many images")
    ap.add_argument("--n_identities", type=int, default=3, help="How many identities to keep")
    ap.add_argument("--out", default="data/team_faces", help="Output folder")
    args = ap.parse_args()

    print("[LFW] Downloading via scikit-learn...")
    lfw = fetch_lfw_people(min_faces_per_person=args.min_per_person, resize=0.5, color=True)
    images = lfw.images / 255.0  # (n, h, w, 3)
    names = np.array(lfw.target_names)[lfw.target]

    uniq, counts = np.unique(names, return_counts=True)
    order = np.argsort(counts)[::-1]
    chosen = uniq[order][:args.n_identities]
    print(f"[LFW] Selected identities: {', '.join(chosen)}")

    for person in chosen:
        outdir = os.path.join(args.out, person.replace(" ", "_"))
        save_person(images, names, person, outdir)

    print(f"[LFW] Saved subset under {args.out}. Next, run alignment:")
    print("  python -m src.preprocessing.align --in data/team_faces --out data/team_faces")

if __name__ == "__main__":
    main()
