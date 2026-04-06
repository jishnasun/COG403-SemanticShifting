import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("/Users/alicebaker/Library/CloudStorage/OneDrive-UniversityofToronto/University/Classes/COG/COG403/Final Project/lexemes.tsv", sep="\t")
glotto = pd.read_csv("/Users/alicebaker/Library/CloudStorage/OneDrive-UniversityofToronto/University/Classes/COG/COG403/Final Project/cldf/languages.csv", sep=",")
glotto.columns = glotto.columns.str.strip()  # remove whitespace issues

# Map language → family/subgroup
lang_to_family = dict(zip(glotto["Name"], glotto["Family"]))
lang_to_subgroup = dict(zip(glotto["Name"], glotto["SubGroup"]))

# Filter accepted semantic shifts
df = df[(df["Status"] == "ACCEPTED")]

# Add family/subgroup columns
df["Family"] = df["Source_Language"].map(lang_to_family)
df["SubGroup"] = df["Source_Language"].map(lang_to_subgroup)

print(f"Total accepted semantic shifts: {len(df)}")

# ===============================
# ENCODE CONCEPTS
# ===============================
concept_encoder = LabelEncoder()
all_concepts = pd.concat([df["Source_Concept"], df["Target_Concept"]])
concept_encoder.fit(all_concepts)

df["source_id"] = concept_encoder.transform(df["Source_Concept"])
df["target_id"] = concept_encoder.transform(df["Target_Concept"])

concept_names = concept_encoder.classes_
n_concepts = len(concept_names)
print(f"Total unique concepts: {n_concepts}")

# ===============================
# CALCULATE GLOBAL / FAMILY / SUBGROUP PRIORS
# ===============================
alpha = 0.5  # smoothing

# Global prior
global_counts = np.zeros(n_concepts)
for t in df["target_id"]:
    global_counts[t] += 1
global_counts += alpha
P_T_global = global_counts / global_counts.sum()

# Family priors
P_T_family = {}
for fam in df["Family"].dropna().unique():
    counts = np.zeros(n_concepts)
    for t in df[df["Family"] == fam]["target_id"]:
        counts[t] += 1
    counts += alpha
    P_T_family[fam] = counts / counts.sum()

# Subgroup priors
P_T_subgroup = {}
for sub in df["SubGroup"].dropna().unique():
    counts = np.zeros(n_concepts)
    for t in df[df["SubGroup"] == sub]["target_id"]:
        counts[t] += 1
    counts += alpha
    P_T_subgroup[sub] = counts / counts.sum()

# ===============================
# CALCULATE P(S|T)
# ===============================
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# paraphrase-multilingual-MiniLM-L12-v2

vectors = model.encode(concept_names)

similarity = cosine_similarity(vectors)
similarity = (similarity + 1) / 2  # normalize 0–1

# neighbor sets
k = 10
neighbor_sets = {i: set(np.argsort(similarity[i])[-k:]) for i in range(n_concepts)}

P_S_given_T = np.zeros_like(similarity)
for t in range(n_concepts):
    for s in range(n_concepts):
        P_S_given_T[s, t] = len(neighbor_sets[t] & neighbor_sets[s])

# normalize columns
col_sums = P_S_given_T.sum(axis=0, keepdims=True)
col_sums[col_sums == 0] = 1
P_S_given_T /= col_sums

# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate_hierarchical_model(df, P_S_given_T, lang_to_family, lang_to_subgroup,
                                P_T_subgroup, P_T_family, P_T_global, threshold=0.05):
    top1_total = 0
    top5_total = 0
    reciprocal_ranks = []
    N = len(df)

    for lang in df["Source_Language"].unique():
        family = lang_to_family.get(lang, None)
        subgroup = lang_to_subgroup.get(lang, None)

        if subgroup in P_T_subgroup:
            P_T = P_T_subgroup[subgroup]
        elif family in P_T_family:
            P_T = P_T_family[family]
        else:
            P_T = P_T_global

        P_T_given_S = P_S_given_T * P_T
        np.fill_diagonal(P_T_given_S, 0)  # set all P(T|S) where T == S to 0

        
        row_sums = P_T_given_S.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P_T_given_S /= row_sums

        df_lang = df[df["Source_Language"] == lang]

        for _, row in df_lang.iterrows():
            s = row["source_id"]
            true_t = row["target_id"]

            probs = P_T_given_S[s]
            ranked_targets = np.argsort(probs)[::-1]

            rank_position = np.where(ranked_targets == true_t)[0][0] + 1

            if rank_position == 1:
                top1_total += 1
            if rank_position <= 5:
                top5_total += 1
            reciprocal_ranks.append(1 / rank_position)

    top1_acc = top1_total / N
    top5_acc = top5_total / N
    mrr = np.mean(reciprocal_ranks)

    return (top1_acc, top5_acc)

    print("\n==============================")
    print("HIERARCHICAL PRIOR MODEL EVALUATION")
    print("==============================")
    print(f"Top-1 Accuracy: {top1_acc:.3f}")
    print(f"Top-5 Accuracy: {top5_acc:.3f}")
    print(f"MRR: {mrr:.3f}")


def run_feval():
    results = evaluate_hierarchical_model(
        df,
        P_S_given_T,
        lang_to_family,
        lang_to_subgroup,
        P_T_subgroup,
        P_T_family,
        P_T_global,
        threshold=0.05
    )
    return results

# ===============================
# RUN EVALUATION
# ===============================
evaluate_hierarchical_model(
    df,
    P_S_given_T,
    lang_to_family,
    lang_to_subgroup,
    P_T_subgroup,
    P_T_family,
    P_T_global,
    threshold=0.05
)
