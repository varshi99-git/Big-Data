import pandas as pd
import numpy as np
import random
import os

# 1. Load the MovieLens 100k data (u.data)
file_path = 'u.data'

if not os.path.exists(file_path):
    print(f"ERROR: '{file_path}' not found in {os.getcwd()}")
    print("Please download it and place it in this folder before running.")
    exit()

# Load only user and movie columns
data = pd.read_csv(file_path, sep='\t', names=['user', 'movie', 'rate', 'ts'])
user_movie_sets = data.groupby('user')['movie'].apply(set).to_dict()

def get_jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

# 2. Compute Exact Similarities >= 0.5 (Ground Truth)
users = list(user_movie_sets.keys())
exact_pairs = set()
print("Calculating exact Jaccard similarities...")
for i in range(len(users)):
    for j in range(i + 1, len(users)):
        u1, u2 = users[i], users[j]
        if get_jaccard(user_movie_sets[u1], user_movie_sets[u2]) >= 0.5:
            exact_pairs.add(tuple(sorted((u1, u2))))

print(f"Total Exact Pairs (Sim >= 0.5): {len(exact_pairs)}")

# 3. Min-Hashing logic
def run_minhash_experiment(num_hashes):
    all_movies = data['movie'].unique()
    max_movie = max(all_movies) + 1
    p = 2039 # Prime larger than 1682 movies
    
    a = random.sample(range(1, p), num_hashes)
    b = random.sample(range(0, p), num_hashes)
    
    signatures = {}
    for user, movies in user_movie_sets.items():
        sig = []
        for h in range(num_hashes):
            min_val = min([(a[h] * m + b[h]) % p for m in movies])
            sig.append(min_val)
        signatures[user] = sig
    
    est_pairs = set()
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1, u2 = users[i], users[j]
            # Estimate similarity by comparing signature components
            match_count = sum(1 for k in range(num_hashes) if signatures[u1][k] == signatures[u2][k])
            if (match_count / num_hashes) >= 0.5:
                est_pairs.add(tuple(sorted((u1, u2))))
    
    fp = len(est_pairs - exact_pairs)
    fn = len(exact_pairs - est_pairs)
    return fp, fn

# 4. Perform 5 runs for each t value as required
t_values = [50, 100, 200]
print("\nRunning Min-Hash Experiments (5 runs per t)...")

for t in t_values:
    fps, fns = [], []
    for run in range(1, 6):
        fp, fn = run_minhash_experiment(t)
        fps.append(fp)
        fns.append(fn)
    
    print(f"\nResults for t = {t}:")
    print(f"  Average False Positives: {np.mean(fps):.2f}")
    print(f"  Average False Negatives: {np.mean(fns):.2f}")