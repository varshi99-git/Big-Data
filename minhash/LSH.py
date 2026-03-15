import math

def calculate_lsh_probability(s, b, r):
    """Calculates the probability of a pair being a candidate."""
    return 1 - (1 - s**r)**b

# Parameters from Step 1
b = 20
r = 8
t = 160  # b * r

# 18 Jaccard Similarity values (from Q1B)
# Example data (Replace these with your actual 18 calculated numbers)
similarities = {
    "D1-D2": 0.77, "D1-D3": 0.25, "D1-D4": 0.33,
    "D2-D3": 0.20, "D2-D4": 0.55, "D3-D4": 0.91
}

print(f"--- LSH Analysis (b={b}, r={r}, threshold≈0.69) ---")
print(f"{'Pair':<10} | {'Jaccard (s)':<12} | {'Prob. of Candidate'}")
print("-" * 45)

for pair, s in similarities.items():
    prob = calculate_lsh_probability(s, b, r)
    print(f"{pair:<10} | {s:<12.4f} | {prob:.6f}")