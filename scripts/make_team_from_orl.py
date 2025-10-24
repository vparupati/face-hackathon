import os, argparse, shutil, glob, re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orl", default="data/att_faces", help="Path to ORL (s1..s40)")
    ap.add_argument("--out", default="data/team_faces", help="Output team faces folder")
    ap.add_argument("--subjects", nargs="*", help="Subjects to use (e.g., s1 s2 s3). Default: first 3 found")
    ap.add_argument("--rename", nargs="*", help="Optional new names for subjects (e.g., Alice Bob Carol)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    subs = args.subjects
    if not subs:
        cand = sorted([d for d in os.listdir(args.orl) if re.match(r"s\d+", d)])
        if len(cand) < 3:
            raise SystemExit("Need at least 3 subjects in ORL.")
        subs = cand[:3]

    names = args.rename if args.rename and len(args.rename)==len(subs) else subs

    for sub, name in zip(subs, names):
        src = os.path.join(args.orl, sub)
        dst = os.path.join(args.out, name)
        os.makedirs(dst, exist_ok=True)
        for p in glob.glob(os.path.join(src, "*")):
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
        print(f"Copied {sub} -> {dst}")

    print(f"Done. Team faces at: {args.out}")

if __name__ == "__main__":
    main()
