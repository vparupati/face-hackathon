import os, re, shutil, glob

src = "data/team_faces/yalefaces"
dst = "data/team_faces"
os.makedirs(dst, exist_ok=True)

for p in glob.glob(os.path.join(src, "*")):
    name = os.path.basename(p)
    m = re.match(r"(subject\d+)", name)
    if not m:
        continue
    person = m.group(1)
    outdir = os.path.join(dst, person)
    os.makedirs(outdir, exist_ok=True)
    shutil.move(p, os.path.join(outdir, name))

try:
    os.rmdir(src)
except OSError:
    pass

print("Reorganized Yale into per-person folders under data/team_faces/")
