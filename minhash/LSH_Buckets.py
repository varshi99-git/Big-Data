import pandas as pd
import numpy as np
import random
import os

# 1. Load Data
file_path = 'u.data'
if not os.path.exists(file_path):
    print("Error: u.data not found.")
    exit()

data = pd.read_csv(file_path, sep='\t', names=['user', 'movie', 'rate', 'ts'])
user_movies = data.groupby('user')['movie'].apply(set).to_dict()
users = list(user_movies.keys())

# 2. Ground Truth for Similarity >= 0.6
print("Calculating ground truth (s >= 0.6)...")
exact_6 = set()
for i in range(len(users)):
    for j in range(i + 1, len(users)):
        u1, u2 = users[i], users[j]
        s1, s2 = user_movies[u1], user_movies[u2]
        if (len(s1 & s2) / len(s1 | s2)) >= 0.6:
            exact_6.add(tuple(sorted((u1, u2))))

# 3. LSH Function
def run_lsh_experiment(t, r, b):
    p = 2039
    a = random.sample(range(1, p), t)
    b_hash = random.sample(range(0, p), t)
    
    # Generate Signatures
    sigs = {}
    for u in users:
        sigs[u] = [min([(a[i]*m + b_hash[i])%p for m in user_movies[u]]) for i in range(t)]
    
    candidates = set()
    # Divide into bands
    for band in range(b):
        buckets = {}
        for u in users:
            # Get the slice of the signature for this band
            band_sig = tuple(sigs[u][band*r : (band+1)*r])
            if band_sig not in buckets:
                buckets[band_sig] = []
            buckets[band_sig].append(u)
        
        # Any users in the same bucket are candidates
        for u_list in buckets.values():
            if len(u_list) > 1:
                for i in range(len(u_list)):
                    for j in range(i + 1, len(u_list)):
                        candidates.add(tuple(sorted((u_list[i], u_list[j]))))
    
    # Calculate False Positives (FP) and False Negatives (FN)
    # FP: Candidate but actual sim < 0.6
    # FN: Actual sim >= 0.6 but not a candidate
    fp = len(candidates - exact_6)
    fn = len(exact_6 - candidates)
    return fp, fn

# 4. Run defined experiments (5 runs each)
configs = [(50, 5, 10), (100, 5, 20), (200, 5, 40), (200, 10, 20)]
print("\n--- LSH Results (Averages over 5 runs) ---")

for t, r, b in configs:
    fps, fns = [], []
    for _ in range(5):
        fp, fn = run_lsh_experiment(t, r, b)
        fps.append(fp)
        fns.append(fn)
    print(f"t={t}, r={r}, b={b} | Avg FP: {np.mean(fps):.2f} | Avg FN: {np.mean(fns):.2f}")