import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ----------------------
# load and clean data
# ----------------------

# load data
df = pd.read_csv("lexemes.tsv", sep="\t")
df = df[(df["Status"] == "ACCEPTED")]

glotto = pd.read_csv("cldf/languages.csv", sep=",")
glotto.columns = glotto.columns.str.strip()

lang_to_family = dict(zip(glotto["Name"], glotto["Family"]))
lang_to_subgroup = dict(zip(glotto["Name"], glotto["SubGroup"]))

df["Family"] = df["Source_Language"].map(lang_to_family)
df["SubGroup"] = df["Source_Language"].map(lang_to_subgroup)


# encode concepts
concept_encoder = LabelEncoder()

all_concepts = pd.concat([df["Source_Concept"], df["Target_Concept"]])
concept_encoder.fit(all_concepts)

df["source_id"] = concept_encoder.transform(df["Source_Concept"])
df["target_id"] = concept_encoder.transform(df["Target_Concept"])

concept_names = concept_encoder.classes_
n_concepts = len(concept_names)


# ----------------------
# model creation
# ----------------------

# calculate priors
alpha = 0.5

# PRIOR - global
global_counts = np.zeros(n_concepts)
for t in df["target_id"]:
    global_counts[t] += 1
global_counts += alpha
P_T_global = global_counts / global_counts.sum()

# PRIOR - family
P_T_family = {}
for fam in df["Family"].dropna().unique():
    counts = np.zeros(n_concepts)
    for t in df[df["Family"] == fam]["target_id"]:
        counts[t] += 1
    counts += alpha
    P_T_family[fam] = counts / counts.sum()

# PRIOR - subgroup
P_T_subgroup = {}
for sub in df["SubGroup"].dropna().unique():
    counts = np.zeros(n_concepts)
    for t in df[df["SubGroup"] == sub]["target_id"]:
        counts[t] += 1
    counts += alpha
    P_T_subgroup[sub] = counts / counts.sum()


# P(S|T) - SEMANTIC SIMILARITY
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

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


# ----------------------
# evaluation function
# ----------------------

# FAMILY EVALUATION
def evaluate_hierarchical_model(df, P_S_given_T, lang_to_family, lang_to_subgroup,
                                P_T_subgroup, P_T_family, P_T_global, threshold=0.05):

    top1_results = []

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

        np.fill_diagonal(P_T_given_S, 0)
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
                top1_results.append(1)
            else:
                top1_results.append(0)

    return top1_results


def run_sig_fam():
    fam1 = evaluate_hierarchical_model(df,
                                       P_S_given_T,
                                       lang_to_family,
                                       lang_to_subgroup,
                                       P_T_subgroup,
                                       P_T_family,
                                       P_T_global,
                                       threshold=0.05)

    return fam1
