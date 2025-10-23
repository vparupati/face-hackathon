import argparse, os, sys, glob
from typing import Tuple
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def process_folder(input_dir: str, output_dir: str, image_size: int = 160, save_landmarks: bool = False):
    mtcnn = MTCNN(image_size=image_size, margin=20, post_process=True, keep_all=False)
    in_paths = sorted([p for p in glob.glob(os.path.join(input_dir, "**", "*")) if p.lower().endswith((".jpg",".jpeg",".png",".bmp"))])
    if not in_paths:
        print(f"No images found under: {input_dir}")
        return

    for src in in_paths:
        try:
            rel = os.path.relpath(src, input_dir)
            out_path = os.path.join(output_dir, rel)
            out_dir = os.path.dirname(out_path)
            ensure_dir(out_dir)

            img = Image.open(src).convert("RGB")
            aligned, probs = mtcnn(img, return_prob=True)
            if aligned is None:
                print(f"[WARN] No face detected: {src}")
                continue

            if isinstance(aligned, list):
                aligned = aligned[0]  # take first face if multiple (hackathon simplification)

            aligned_img = Image.fromarray((aligned.permute(1,2,0).numpy() * 255).astype(np.uint8))
            aligned_img.save(out_path)

            if save_landmarks and hasattr(mtcnn, "landmarks") and mtcnn.landmarks is not None:
                # facenet-pytorch MTCNN returns landmarks internally; exposing here is simplified
                pass
        except Exception as e:
            print(f"[ERROR] {src}: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input root with raw images in class subfolders")
    ap.add_argument("--out", dest="out", required=True, help="Output root for aligned 160x160 crops")
    ap.add_argument("--size", dest="size", type=int, default=160, help="Output image size")
    ap.add_argument("--save-landmarks", action="store_true", help="(Optional) Save landmarks if available")
    args = ap.parse_args()
    process_folder(args.inp, args.out, image_size=args.size, save_landmarks=args.save_landmarks)
