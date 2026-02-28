import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# data retrieval + get teh accepted shifts only
df = pd.read_csv(
    "C:/Users/starw/PycharmProjects/COG403Project/cldf/lexemes.tsv",
    sep="\t"
)

df = df[(df["Status"] == "ACCEPTED") & (df["Direction"] == "→")]
print("Total accepted directional shifts:", len(df))

# encode concepts for faster identification during model computation
concept_encoder = LabelEncoder()

all_concepts = pd.concat([df["Source_Concept"], df["Target_Concept"]])
concept_encoder.fit(all_concepts)

df["source_id"] = concept_encoder.transform(df["Source_Concept"])
df["target_id"] = concept_encoder.transform(df["Target_Concept"])

n_concepts = len(concept_encoder.classes_)
print("Total unique concepts:", n_concepts)

# global counts to get priors for bayesian calculation
# this was before talking to prof
global_counts = np.zeros((n_concepts, n_concepts))

for row in df.itertuples():
    global_counts[row.source_id, row.target_id] += 1

# bayesian posterior
alpha = 0.5  # weak Dirichlet prior according to chat

global_posterior = global_counts + alpha
global_transition_probs = global_posterior / global_posterior.sum(axis=1, keepdims=True)

# Language specific counts / frequency
transition_counts_lang = defaultdict(
    lambda: defaultdict(lambda: defaultdict(int))
)

for row in df.itertuples():
    lang = row.Source_Language
    s = row.source_id
    t = row.target_id
    transition_counts_lang[lang][s][t] += 1

print(f"Number of languages: {len(transition_counts_lang)}")

# hierarchal smoothing, basically normalizes data
kappa = 5  # strength of global prior,, essentially a weighting
threshold = 0.25  # can change this, basically only prints anything with a probability over 0.25

for lang in transition_counts_lang:
    print("\n==============================")  # separators, I saw them online and liked how they look lmao
    print("Language:", lang)
    print("==============================")

    for s in transition_counts_lang[lang]:

        # Global prior for this source concept
        prior = kappa * global_transition_probs[s]

        # copy prior to posterior
        posterior = prior.copy()

        # Add language-specific counts
        for t in transition_counts_lang[lang][s]:
            posterior[t] += transition_counts_lang[lang][s][t]

        # Normalize
        total = posterior.sum()
        if total > 0:
            posterior /= total

        for t, prob in enumerate(posterior):
            if prob > threshold:
                source = concept_encoder.inverse_transform([s])[0]
                target = concept_encoder.inverse_transform([t])[0]
                print(f"{source} → {target} : {round(prob, 3)}")
