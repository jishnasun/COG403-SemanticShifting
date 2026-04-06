import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("cldf/lexemes.tsv", sep="\t")
glotto = pd.read_csv("cldf/languages.csv", sep=",")
glotto.columns = glotto.columns.str.strip()  # remove whitespace issues

# Map language → family/subgroup
lang_to_family = dict(zip(glotto["Name"], glotto["Family"]))
lang_to_subgroup = dict(zip(glotto["Name"], glotto["SubGroup"]))

# Filter accepted semantic shifts
df = df[(df["Status"] == "ACCEPTED")]
# & (df["Direction"] == "→")

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
# paraphrase-multilingual-mpnet-base-v2

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


# =====================================================
# NEW: BUILD P(T|S)
# =====================================================

def normalize_rows(P):
    np.fill_diagonal(P, 0)  # set all P(T|S) where T == S to 0
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return P / row_sums

# GLOBAL MODEL
P_global = normalize_rows(P_S_given_T * P_T_global)

# FAMILY MODEL
P_family = np.zeros_like(P_S_given_T)
for fam in P_T_family:
    fam_sources = df[df["Family"] == fam]["source_id"].unique()
    for s in fam_sources:
        P_family[s] = P_S_given_T[s] * P_T_family[fam]
P_family = normalize_rows(P_family)

# SUBGROUP MODEL  ⭐ strongest model
P_subgroup = np.zeros_like(P_S_given_T)
for sub in P_T_subgroup:
    sub_sources = df[df["SubGroup"] == sub]["source_id"].unique()
    for s in sub_sources:
        P_subgroup[s] = P_S_given_T[s] * P_T_subgroup[sub]
P_subgroup = normalize_rows(P_subgroup)


# =====================================================
# GLOBAL EVALUATION
# =====================================================

def evaluate_subset(P_matrix, df_subset):

    top1_correct = 0
    top5_correct = 0
    reciprocal_ranks = []

    for _, row in df_subset.iterrows():

        s = row["source_id"]
        true_t = row["target_id"]

        probs = P_matrix[s]
        ranked_targets = np.argsort(probs)[::-1]

        rank_position = np.where(ranked_targets == true_t)[0][0] + 1

        if rank_position == 1:
            top1_correct += 1
        if rank_position <= 5:
            top5_correct += 1

        reciprocal_ranks.append(1 / rank_position)

    N = len(df_subset)

    return {
        "top1": top1_correct / N,
        "top5": top5_correct / N,
        "mrr": np.mean(reciprocal_ranks)
    }


def evaluate_model(P_matrix, label):
    print("\n==============================")
    print("EVALUATING:", label)
    print("==============================")

    metrics = evaluate_subset(P_matrix, df)

    print("Top-1 Accuracy:", round(metrics["top1"], 3))
    print("Top-5 Accuracy:", round(metrics["top5"], 3))
    print("MRR:", round(metrics["mrr"], 3))


# =====================================================
# RUN MODELS
# =====================================================

evaluate_model(P_global,   "GLOBAL PRIOR MODEL")
evaluate_model(P_family,   "FAMILY PRIOR MODEL")
evaluate_model(P_subgroup, "SUBGROUP PRIOR MODEL")


language_results = []

for lang in df["Source_Language"].unique():

    df_lang = df[df["Source_Language"] == lang]
    if len(df_lang) < 5:
        continue

    metrics = evaluate_subset(P_subgroup, df_lang)

    language_results.append({
        "language": lang,
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "mrr": metrics["mrr"],
        "N": len(df_lang)
    })

# =====================================================
# FUNCTION TO GET SHIFT DETAILS
# =====================================================
def get_shift_results(P_matrix, df_subset):
    """
    Returns a DataFrame with each shift and whether it was correctly predicted
    in the top-1.
    """
    results = []
    for _, row in df_subset.iterrows():
        s = row["source_id"]
        true_t = row["target_id"]
        source_concept = row["Source_Concept"]
        target_concept = row["Target_Concept"]

        probs = P_matrix[s]
        ranked_targets = np.argsort(probs)[::-1]
        top1_pred = ranked_targets[0]

        correct = (top1_pred == true_t)

        results.append({
            "source_concept": source_concept,
            "target_concept": target_concept,
            "predicted_top1": concept_names[top1_pred],
            "correct_top1": correct
        })
    return pd.DataFrame(results)


# =====================================================
# LANGUAGE-LEVEL EVALUATION WITH SHIFT DETAILS
# =====================================================
language_results = []

for lang in df["Source_Language"].unique():
    df_lang = df[df["Source_Language"] == lang]
    if len(df_lang) < 5:
        continue

    metrics = evaluate_subset(P_subgroup, df_lang)

    language_results.append({
        "language": lang,
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "mrr": metrics["mrr"],
        "N": len(df_lang),
        "shifts_df": get_shift_results(P_subgroup, df_lang)
    })

# Sort by top1 first, then top5 as tie-breaker
lang_df = pd.DataFrame(language_results)
lang_df = lang_df.sort_values(by=["top1", "top5"], ascending=[False, False])

best_lang = lang_df.iloc[0]
worst_lang = lang_df.iloc[-1]

# =====================================================
# PRINT LANGUAGE RANKING
# =====================================================
print("\n==============================")
print("SUBGROUP PRIOR MODEL — LANGUAGE RANKING")
print("==============================")
print(lang_df[["language", "top1", "top5", "mrr", "N"]])

# =====================================================
# BEST LANGUAGE
# =====================================================
print("\nBEST LANGUAGE:", best_lang["language"])
print(best_lang[["top1", "top5", "mrr", "N"]])
print("\nShifts for BEST language with prediction info:")
print(best_lang["shifts_df"])

# =====================================================
# WORST LANGUAGE
# =====================================================
print("\nWORST LANGUAGE:", worst_lang["language"])
print(worst_lang[["top1", "top5", "mrr", "N"]])
print("\nShifts for WORST language with prediction info:")
print(worst_lang["shifts_df"])

# UNCOMMENT TO SAVE CSV OF RANKED LANGUAGES
# lang_df.to_csv("language_ranking_glotto.csv", index=False)


# =====================================================
# FOR VISUALIZATION
# =====================================================
df_masai = lang_df[lang_df["language"] == "Masai"]
df_yoruba = lang_df[lang_df["language"] == "Yoruba"]


def get_best_full():
    return tuple(df_masai[['top1', 'top5']].iloc[0])


def get_best_glotto():
    return tuple(best_lang[['top1', 'top5']])


def get_worst_fam():
    return tuple(df_yoruba[['top1', 'top5']].iloc[0])
