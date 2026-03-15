import itertools
import os

# Your existing functions
def get_k_grams(text, k, mode='char'):
    """Constructs unique k-grams for characters or words."""
    if mode == 'word':
        tokens = text.split()
    else:
        tokens = text
    
    # Generate the k-grams
    grams = [tokens[i:i+k] for i in range(len(tokens) - k + 1)]
    
    # Format word k-grams as strings for easy set operations
    if mode == 'word':
        grams = [" ".join(g) for g in grams]
        
    return set(grams) # [cite: 19] Store each only once; duplicates ignored.

def jaccard_similarity(set_a, set_b):
    """Calculates Jaccard Similarity: Intersection / Union."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

# --- Main Logic for Question 1 ---

# 1. Define filenames and check existence
doc_names = ['D1.txt', 'D2.txt', 'D3.txt', 'D4.txt']
docs = {}

print("--- Checking for Files ---")
for filename in doc_names:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            # [cite: 14] All documents have lowercase letters and space.
            # We use .lower() to ensure consistency.
            key = filename.replace('.txt', '')
            docs[key] = f.read().lower().strip()
            print(f"Loaded {filename} successfully.")
    else:
        print(f"CRITICAL ERROR: {filename} not found in {os.getcwd()}")

# 2. Only proceed if all 4 documents loaded correctly
if len(docs) < 4:
    print("\nStop! Please put D1.txt, D2.txt, D3.txt, and D4.txt in this folder.")
else:
    # Define the required k-gram types [cite: 16, 17, 18]
    configs = [
        ('Character 2-grams', 2, 'char'),
        ('Character 3-grams', 3, 'char'),
        ('Word 2-grams', 2, 'word')
    ]

    # All unique pairs of documents (6 pairs total)
    pairs = list(itertools.combinations(['D1', 'D2', 'D3', 'D4'], 2))

    print("\n--- Part 1B: Jaccard Similarities (18 Numbers) ---")
    for label, k, mode in configs:
        print(f"\nType: {label}")
        for d1, d2 in pairs:
            set1 = get_k_grams(docs[d1], k, mode)
            set2 = get_k_grams(docs[d2], k, mode)
            sim = jaccard_similarity(set1, set2)
            print(f"{d1} and {d2}: {sim:.4f}")