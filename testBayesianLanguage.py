import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

'''
NOTES!!!
- I put TODOs where ever there are variables we can adjust / fiddle with
- Might need to install some of those packages up there
    - On that note, sentence_transformers is a bit finicky so try pip install if the packages tab doesn't work
- later we can separate it and do a version without P(T), but for now lets just submit one file for feasability
- I mention it later but we pull from a pretrained language model for semantic relatedness
    - might need to change this depending on what the prof says, it seems to work though
- We should probably come up with some sort of visualization but that's for future us
    - A next step could maybe be exporting the data to a csv or tsv or txt file to use later for visualization
    - might be a pain if we're adjusting the variables though
- We can also add an additional prior using the glottolog for language families
'''

# Retrieve data from lexemes.tsv (change the file path depending on your computer)
df = pd.read_csv("C:/Users/starw/PycharmProjects/COG403Project/cldf/lexemes.tsv", sep="\t")

# take a look at lexemes.tsv but there's basically a bunch of columns with SOURCE, TARGET
# and the direction of semantic shift. It's raw so not all of them are actual shifts
df = df[(df["Status"] == "ACCEPTED") & (df["Direction"] == "→")]  # pick out the actual accepted shifts

print(f"Total accepted semantic shifts: {len(df)}")  # can comment this out, mostly for testing


# Encoding concepts using LabelEncoder because it speeds up identification for model computation according to google
# There are source and target id columns but those are basically strings so not much of a difference, ignore them
concept_encoder = LabelEncoder()

all_concepts = pd.concat([df["Source_Concept"], df["Target_Concept"]])
concept_encoder.fit(all_concepts)

df["source_id"] = concept_encoder.transform(df["Source_Concept"])
df["target_id"] = concept_encoder.transform(df["Target_Concept"])

concept_names = concept_encoder.classes_
n_concepts = len(concept_names)

print(f"Total unique concepts: {n_concepts}")  # again can comment out, but I like having it for reference

'''
!!! CALCULATING P(T) !!!
this is just the global frequency count
'''

# TODO: Adjust alpha as needed
alpha = 0.5  # this is that same weak Dirichlet prior which makes it so nothing is absolutely 0 to smoothen data
# and prevent overfitting! Basically everything auto starts at 0.5 instead of 0
# Can play around with this later to see what level of smoothening is best (higher alpha = more smoothening)

target_counts = np.zeros(n_concepts)
for t in df["target_id"]:
    target_counts[t] += 1

target_counts += alpha

P_T = target_counts / target_counts.sum()

'''
!!! CALCULATING P(S|T) !!!
this uses feature-overlap like in the lexical creativity paper for semantic relatedness
'''

model = SentenceTransformer("all-MiniLM-L6-v2")  # stealing form a pretrained language model for vector embeddings

vectors = model.encode(concept_names)

# cosine similarity of vectors
similarity = cosine_similarity(vectors)
similarity = (similarity + 1) / 2  # making it so 0 = less similar and 1 = more similar

# neighbor sets for feature approximation
# TODO: Adjust K as needed
k = 10  # basically k's nearest neighbour model for relatedness, can also fiddle with this
neighbor_sets = {
    i: set(np.argsort(similarity[i])[-k:])  # making a set for easier comparison, csc384 assignment 1 trauma
    for i in range(n_concepts)
}

# probability according to feature overlap -- basically the more shared neighbours the more semantically related
P_S_given_T = np.zeros_like(similarity)

for t in range(n_concepts):
    for s in range(n_concepts):
        overlap = len(neighbor_sets[t] & neighbor_sets[s])
        P_S_given_T[s, t] = overlap

# normalize columns for P(S|T), same as the normalization from above, especially to avoid dividing by 0
col_sums = P_S_given_T.sum(axis=0, keepdims=True)
col_sums[col_sums == 0] = 1
P_S_given_T = P_S_given_T / col_sums

# sharpening -- basically can make the distribution more extreme, don't need this but worth looking into
# TODO: Adjust beta as needed
# beta = 3
# P_S_given_T = np.power(P_S_given_T, beta)
# P_S_given_T /= P_S_given_T.sum(axis=0, keepdims=True)  # re-normalizing after sharpening

'''
!!! FINAL BAYES EQUATION !!!
easiest part
'''

# TODO: Adjust threshold as needed
threshold = 0.05  # again so we aren't printing everything, just where there's an actual probability

P_T_given_S = P_S_given_T * P_T

# normalize rows (so the total is always 1)
row_sums = P_T_given_S.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
P_T_given_S = P_T_given_S / row_sums


# printing results by language:
languages = df["Source_Language"].unique()

for lang in languages:

    print("\n======================================")  # divider, can use to search results to skip between languages
    print(f"LANGUAGE: {lang}")
    print("======================================")

    df_lang = df[df["Source_Language"] == lang]

    if len(df_lang) == 0:
        continue

    # only include sources that actually appear in this language
    source_ids = set(df_lang["source_id"])

    for s in source_ids:

        strong = [
            (t, P_T_given_S[s, t])
            for t in range(n_concepts)
            if P_T_given_S[s, t] > threshold and t != s
        ]

        if strong:
            source = concept_names[s]
            print(f"\nSource: {source}")

            for t, prob in sorted(strong, key=lambda x: -x[1])[:10]:
                target = concept_names[t]
                print(f"  → {target} : {round(prob, 3)}")
