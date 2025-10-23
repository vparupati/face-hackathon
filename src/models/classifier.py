import argparse, joblib, os, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds", required=True, help="NPZ with embeddings, labels, paths")
    ap.add_argument("--out", required=True, help="Output .joblib")
    args = ap.parse_args()

    data = np.load(args.embeds, allow_pickle=True)
    X = data["embeddings"]; y = data["labels"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # kNN
    knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    knn.fit(Xtr, ytr)
    p_knn = knn.predict(Xte)
    acc_knn = accuracy_score(yte, p_knn)

    # LinearSVC (on cosine-normalized embeddings works fine)
    svc = LinearSVC()
    svc.fit(Xtr, ytr)
    p_svc = svc.predict(Xte)
    acc_svc = accuracy_score(yte, p_svc)

    if acc_knn >= acc_svc:
        best = ("knn", knn, acc_knn)
    else:
        best = ("svc", svc, acc_svc)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"type": best[0], "model": best[1]}, args.out)
    print(f"Saved recognizer: {args.out}  ({best[0]} acc={best[2]:.3f})")

if __name__ == "__main__":
    main()
