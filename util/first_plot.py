import pandas as pd
import matplotlib.pyplot as plt
import os

# file path
file_path = "/work3/s233559/Thesis/results/judge/llama8/judge_llama8_generator_llama8_oneshot_helueval_results.jsonl"

# load jsonl
df = pd.read_json(file_path, lines=True)
# normalize text columns
df["prediction"] = df["prediction"].astype(str).str.strip().str.upper()
df["ground_truth"] = df["ground_truth"].astype(str).str.strip().str.upper()
df["generators_answer"] = df["generators_answer"].astype(str).str.strip().str.upper()

# check generator correctness
df["generator_correct"] = df["generators_answer"] == df["ground_truth"]

# classify judge behaviour
def classify(row):
    if row["generator_correct"] and row["prediction"] == "AGREE":
        return "Correct Agree"
    elif (not row["generator_correct"]) and row["prediction"] == "DISAGREE":
        return "Correct Disagree"
    elif (not row["generator_correct"]) and row["prediction"] == "AGREE":
        return "Wrong Agree"
    else:
        return "Wrong Disagree"

df["judge_case"] = df.apply(classify, axis=1)

# count categories
counts = df["judge_case"].value_counts()

order = ["Correct Agree", "Correct Disagree", "Wrong Agree", "Wrong Disagree"]
counts = counts.reindex(order)

# create save directory
save_dir = "/work3/s233559/Thesis/results/plots"
os.makedirs(save_dir, exist_ok=True)

# plot
plt.figure(figsize=(8,6))
counts.plot(kind="bar")

plt.title("Judge Diagnostic Categories judge_llama8_generator_llama8")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=30)

plt.tight_layout()

# save
save_path = os.path.join(save_dir, "judge_diagnostic_judge_llama8_generator_llama8.png")
plt.savefig(save_path, dpi=300)

plt.show()

print(f"Plot saved at: {save_path}")