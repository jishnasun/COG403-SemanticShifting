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

# P(T) - PRIOR
alpha = 0.5

target_counts = np.zeros(n_concepts)
for t in df["target_id"]:
    target_counts[t] += 1

target_counts += alpha
P_T = target_counts / target_counts.sum()


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


# BUILD P(T|S)
def build_P_T_given_S(mode):

    if mode == "full":
        # print(f'single: p_s_t = {P_S_given_T}, p_t - {P_T}')
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


# building the P(T|S)
p_full = build_P_T_given_S("full")
p_prior = build_P_T_given_S("prior")
p_sim = build_P_T_given_S("similarity")


# ----------------------
# evaluation function
# ----------------------

# GLOBAL EVALUATION
def evaluate_model(P_T_given_S):
    top1_results = []

    for _, row in df.iterrows():

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


def run_sig_global():
    full1 = evaluate_model(p_full)
    prior1 = evaluate_model(p_prior)
    sim1 = evaluate_model(p_sim)

    return full1, prior1, sim1
