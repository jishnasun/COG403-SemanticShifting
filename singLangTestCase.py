import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from testCaseGlotto import get_best_full, get_worst_fam, get_best_glotto



# =====================================================
# LOAD + FILTER DATA
# =====================================================

df = pd.read_csv("cldf/lexemes.tsv", sep="\t")

df = df[(df["Status"] == "ACCEPTED")]
#  & (df["Direction"] == "→")
print("Total accepted semantic shifts:", len(df))


# =====================================================
# ENCODE CONCEPTS
# =====================================================

concept_encoder = LabelEncoder()
all_concepts = pd.concat([df["Source_Concept"], df["Target_Concept"]])
concept_encoder.fit(all_concepts)

df["source_id"] = concept_encoder.transform(df["Source_Concept"])
df["target_id"] = concept_encoder.transform(df["Target_Concept"])

concept_names = concept_encoder.classes_
n_concepts = len(concept_names)

print("Total unique concepts:", n_concepts)


# =====================================================
# P(T)  — PRIOR
# =====================================================

alpha = 0.5

target_counts = np.zeros(n_concepts)
for t in df["target_id"]:
    target_counts[t] += 1

target_counts += alpha
P_T = target_counts / target_counts.sum()


# =====================================================
# P(S|T) — SEMANTIC SIMILARITY
# =====================================================

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# paraphrase-multilingual-MiniLM-L12-v2
# paraphrase-multilingual-mpnet-base-v2

vectors = model.encode(concept_names)

similarity = cosine_similarity(vectors)
similarity = (similarity + 1) / 2

k = 10
neighbor_sets = {
    i: set(np.argsort(similarity[i])[-k:])
    for i in range(n_concepts)
}

P_S_given_T = np.zeros_like(similarity)

for t in range(n_concepts):
    for s in range(n_concepts):
        overlap = len(neighbor_sets[t] & neighbor_sets[s])
        P_S_given_T[s, t] = overlap

col_sums = P_S_given_T.sum(axis=0, keepdims=True)
col_sums[col_sums == 0] = 1
P_S_given_T = P_S_given_T / col_sums


# =====================================================
# BUILD P(T|S)
# =====================================================

def build_P_T_given_S(mode):

    if mode == "full":
        P = P_S_given_T * P_T

    elif mode == "prior":
        P = np.tile(P_T, (n_concepts, 1))

    elif mode == "similarity":
        P = P_S_given_T.copy()

    else:
        raise ValueError("Unknown mode")

    np.fill_diagonal(P, 0)  # set all P(T|S) where T == S to 0

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return P / row_sums


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

P_full  = build_P_T_given_S("full")

evaluate_model(P_full,  "FULL MODEL  P(T|S) ∝ P(S|T)P(T)")


# =====================================================
# BEST & WORST LANGUAGE — FULL MODEL ONLY WITH SHIFT DETAILS
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


language_results = []

for lang in df["Source_Language"].unique():
    df_lang = df[df["Source_Language"] == lang]
    if len(df_lang) < 5:
        continue

    metrics = evaluate_subset(P_full, df_lang)

    language_results.append({
        "language": lang,
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "mrr": metrics["mrr"],
        "N": len(df_lang),
        "shifts_df": get_shift_results(P_full, df_lang)
    })

lang_df = pd.DataFrame(language_results)
lang_df = lang_df.sort_values(by=["top1", "top5"], ascending=[False, False])

best_lang = lang_df.iloc[0]
worst_lang = lang_df.iloc[-1]

print("\n==============================")
print("FULL MODEL — LANGUAGE RANKING")
print("==============================")
print(lang_df[["language", "top1", "top5", "mrr", "N"]])

print("\nBEST LANGUAGE (FULL MODEL):", best_lang["language"])
print(best_lang[["top1", "top5", "mrr", "N"]])
print("\nShifts for BEST language with prediction info:")
print(best_lang["shifts_df"])

# UNCOMMENT TO SEE SHIFTS FOR SPECIFIC LANGUAGE (according to rank)
# print("\nSING. LANGUAGE SHIFTS (FULL MODEL):")
# curr_lang = lang_df.iloc[1]
# print(curr_lang["shifts_df"])

print("\nWORST LANGUAGE (FULL MODEL):", worst_lang["language"])
print(worst_lang[["top1", "top5", "mrr", "N"]])
print("\nShifts for WORST language with prediction info:")
print(worst_lang["shifts_df"])

# UNCOMMENT TO SAVE CSV OF RANKED LANGUAGES
# lang_df.to_csv("language_ranking_full.csv", index=False)

# =====================================================
# FOR VISUALIZATION
# =====================================================
def get_best_global():
    # get language
    lang = best_lang["language"]
    df_blang = df[df["Source_Language"] == lang]

    # get results
    Mprior = evaluate_subset(P_prior, df_blang)
    Msim = evaluate_subset(P_sim, df_blang)

    # compile results
    full = tuple(best_lang[["top1", "top5"]])
    prior = (Mprior["top1"], Mprior["top5"])
    sim = (Msim["top1"], Msim["top5"])
    fam = get_best_full()

    results = {
        'language': lang,
        'full': full,
        'prior': prior,
        'sim': sim,
        'fam': fam
    }

    return results


def get_best_fam():
    lang = 'Standard Zhuang'
    df_flang = df[df["Source_Language"] == lang]

    # get results
    Ffull = evaluate_subset(P_full, df_flang)
    Fprior = evaluate_subset(P_prior, df_flang)
    Fsim = evaluate_subset(P_sim, df_flang)

    full = (Ffull["top1"], Ffull["top5"])
    prior = (Fprior["top1"], Fprior["top5"])
    sim = (Fsim["top1"], Fsim["top5"])
    fam = get_best_glotto()

    results = {
        'language': lang,
        'full': full,
        'prior': prior,
        'sim': sim,
        'fam': fam
    }

    return results


def get_worst():
    lang = worst_lang["language"]
    df_wlang = df[df["Source_Language"] == lang]

    # get results
    Mprior = evaluate_subset(P_prior, df_wlang)
    Msim = evaluate_subset(P_sim, df_wlang)

    # compile results
    full = tuple(worst_lang[["top1", "top5"]])
    prior = (Mprior["top1"], Mprior["top5"])
    sim = (Msim["top1"], Msim["top5"])
    fam = get_worst_fam()

    results = {
        'language': lang,
        'full': full,
        'prior': prior,
        'sim': sim,
        'fam': fam
    }

    return results
