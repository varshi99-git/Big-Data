import random
import os

# --- Step 1: Define the function correctly ---
def get_char_grams(text, k):
    """
    Constructs unique k-grams based on characters.
    Spaces count as characters[cite: 15]. 
    Duplicates are ignored[cite: 19].
    """
    # Create the sliding window of characters
    grams = [text[i:i+k] for i in range(len(text) - k + 1)]
    return set(grams) # Using set() ensures duplicates are ignored [cite: 19]

def build_minhash_signature(k_gram_set, t, m=10001):
    """
    Builds a min-hash signature using t hash functions[cite: 36, 38].
    m should be greater than 10,000[cite: 34].
    """
    # Convert k-grams to numbers for the hash formula
    hashed_grams = [hash(g) for g in k_gram_set]
    
    random.seed(42) # Fixed seed for consistent results
    a_coeffs = random.sample(range(1, m), t)
    b_coeffs = random.sample(range(0, m), t)
    
    signature = []
    for i in range(t):
        min_val = float('inf')
        for x in hashed_grams:
            # Formula: h(x) = (ax + b) % m
            h = (a_coeffs[i] * x + b_coeffs[i]) % m
            if h < min_val:
                min_val = h
        signature.append(min_val)
    return signature

# --- Step 2: Load the documents ---
# Ensure D1.txt and D2.txt are in the same folder as this script
docs = {}
for doc_id in ['D1', 'D2']:
    filename = f"{doc_id}.txt"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            docs[doc_id] = f.read().lower().strip()
    else:
        print(f"Error: {filename} not found in the current directory.")

# --- Step 3: Run the logic ---
if 'D1' in docs and 'D2' in docs:
    # This line was causing your error; now get_char_grams is defined above
    d1_3grams = get_char_grams(docs['D1'], 3) 
    d2_3grams = get_char_grams(docs['D2'], 3)

    # Required t values: 20, 60, 150, 300, 600 [cite: 22, 37]
    t_values = [20, 60, 150, 300, 600]

    print("--- Min-Hash Results for D1 and D2 ---")
    for t in t_values:
        sig1 = build_minhash_signature(d1_3grams, t)
        sig2 = build_minhash_signature(d2_3grams, t)
        
        # Estimate Similarity = matching components / total components [cite: 25]
        matches = sum(1 for i, j in zip(sig1, sig2) if i == j)
        sim = matches / t
        print(f"t={t}: Estimated Jaccard Similarity = {sim:.4f}")