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

# Retrieve data from lexemes.tsv and languages.csv (change the file path depending on your computer)
df = pd.read_csv("C:/Users/starw/PycharmProjects/COG403Project/cldf/lexemes.tsv", sep="\t")
glotto = pd.read_csv("C:/Users/starw/PycharmProjects/COG403Project/cldf/languages.csv", sep=",")

# 'Name' == 'Source_Language'
lang_to_family = dict(zip(glotto["Name"], glotto["Family"]))  # less specific
lang_to_subgroup = dict(zip(glotto["Name"], glotto["SubGroup"]))  # more specific

# take a look at lexemes.tsv but there's basically a bunch of columns with SOURCE, TARGET
# and the direction of semantic shift. It's raw so not all of them are actual shifts
df = df[(df["Status"] == "ACCEPTED") & (df["Direction"] == "→")]  # pick out the actual accepted shifts

# Map Family and Subgroup to Source_Language using Name
df["Family"] = df["Source_Language"].map(lang_to_family)
df["SubGroup"] = df["Source_Language"].map(lang_to_subgroup)

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
alpha = 0.5  # higher alpha = more smoothening

# Global counts
global_counts = np.zeros(n_concepts)
for t in df["target_id"]:
    global_counts[t] += 1
global_counts += alpha
P_T_global = global_counts / global_counts.sum()

families = df["Family"].dropna().unique()
P_T_family = {}

# Family counts
for fam in families:
    df_fam = df[df["Family"] == fam]

    counts = np.zeros(n_concepts)
    for t in df_fam["target_id"]:
        counts[t] += 1

    counts += alpha
    P_T_family[fam] = counts / counts.sum()

# Subgroup counts
subgroups = df["SubGroup"].dropna().unique()
P_T_subgroup = {}

for sub in subgroups:
    df_sub = df[df["SubGroup"] == sub]

    counts = np.zeros(n_concepts)
    for t in df_sub["target_id"]:
        counts[t] += 1

    counts += alpha
    P_T_subgroup[sub] = counts / counts.sum()

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

# TODO: Adjust threshold as needed
threshold = 0.05  # again so we aren't printing everything, just where there's an actual probability

# printing results by language:
languages = df["Source_Language"].unique()

for lang in languages:
    # Calculate P(T) for specific language (Fallback on less specific counts for smaller subgroups/families)
    family = lang_to_family.get(lang, None)
    subgroup = lang_to_subgroup.get(lang, None)

    if subgroup in P_T_subgroup:
        P_T = P_T_subgroup[subgroup]
    elif family in P_T_family:
        P_T = P_T_family[family]
    else:
        P_T = P_T_global

    '''FINAL BAYES EQUATION'''
    P_T_given_S = P_S_given_T * P_T
    # Normalize
    row_sums = P_T_given_S.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P_T_given_S = P_T_given_S / row_sums

    '''
    # TODO: tune these later
    lambda_sub = 0.5
    lambda_fam = 0.3
    lambda_global = 0.2

    if subgroup is not None:
        P_T = (
                lambda_sub * P_T_subgroup[subgroup] +
                lambda_fam * P_T_family[family] +
                lambda_global * P_T_global
        )
    else:
        # fallback if subgroup unknown
        P_T = (
                0.7 * P_T_family[family] +
                0.3 * P_T_global
        )
        '''

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
