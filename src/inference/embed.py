import os, argparse, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.siamese import Encoder

class FolderDataset(Dataset):
    def __init__(self, root: str, img_size: int = 160):
        self.samples = []
        self.labels = []
        self.paths = []
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
        for ci, c in enumerate(classes):
            cdir = os.path.join(root, c)
            for p in glob.glob(os.path.join(cdir, "*")):
                if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".pgm")):
                    self.samples.append(p)
                    self.labels.append(c)
                    self.paths.append(p)
        self.tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img = Image.open(self.samples[i]).convert("RGB")
        return self.tf(img), self.labels[i], self.paths[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input folder: class subfolders")
    ap.add_argument("--encoder", required=True, help="Path to encoder .pt")
    ap.add_argument("--out", required=True, help="Output .npz")
    ap.add_argument("--img", type=int, default=160)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FolderDataset(args.inp, img_size=args.img)
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    model = Encoder()
    model.load_state_dict(torch.load(args.encoder, map_location=device))
    model.to(device); model.eval()

    all_z, all_y, all_p = [], [], []
    with torch.no_grad():
        for x, ys, ps in ld:
            x = x.to(device)
            z = model(x).cpu().numpy()
            all_z.append(z); all_y.extend(ys); all_p.extend(ps)

    embeds = np.concatenate(all_z, axis=0)
    labels = np.array(all_y)
    paths = np.array(all_p)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, embeddings=embeds, labels=labels, paths=paths)
    print(f"Saved: {args.out}  shapes: embeds={embeds.shape}, labels={labels.shape}")

if __name__ == "__main__":
    main()
