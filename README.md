# COG403-SemanticShifting
All code used for COG403 project on Bayesian modelling of semantic shift across different languages.
<br>

`cldf` folder contains data from DatSemShift database (https://datsemshift.ru/). Relevant files are listed below:

| File                                  | Description                                                                                                                        |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `lexemes.tsv`                             | Semantic shift data, including source and target concepts, language, and acceptability of shift.                      |
| `languages.csv`  | Language data, including language family and subgroup, and glottolog ID.                                                |  
<br>

`language_rankings` folder contains list of languages (with ≥5 accepted shifts) ranked by model accuracy. Includes the language name, Top-1 accuracy, Top-5 accuracy, MMR, number of accepted shifts (*n*), and list of *n* shifts labelled as correctly or incorrectly predicted:

| File                                  | Description                                                                                                                        |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `language_ranking_full.csv`                             | Languages ranked by accuracy of `BayesianLanguage_model.py`.                      |
| `language_ranking_glotto.csv`  | Languages ranked by accuracy of `BayesianGlotto_model.py`.                                                 |
<br>

Relevant code regarding models, evaluation, and visualization of results:

| File                                  | Description                                                                                                                        |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `BayesianLanguage_model.py`                             | Proposed bayesian model predicting semantic shift target concept, given the source concept.                   |
| `BayesianGlotto_model.py`  | Modified bayesian model predicting semantic shift target concept, given the source concept. Prior accounts for language family of shift.                                                 |
| `test_models.py`  | Evaluates proposed bayesian model (`BayesianLanguage_model.py`) and its constituents. Provides overall Top-1 and Top-5 model accuracy.                                                |
| `test_glottoModel.py`  | Evaluates modified bayesian model (`BayesianGlotto_model.py`). Provides overall Top-1 and Top-5 model accuracy.                                                 |
| `singLangTestCase.py`  | Evaluates proposed bayesian model accuracy per language and ranks them to produce `language_ranking_full.csv`.                                                 |
| `testCaseGlotto.py`  | Evaluates modified bayesian model accuracy per language and ranks them to produce `language_ranking_glotto.csv`.                                                 |
| `results.py`  | Visualization of accuracy results for all models and constituents.                                                 |
| `test_significant.py`  | [description]                                                 |
| `test_significant_fam.py`  | [description]                                                 |
| `significanceTesting.py`  | [description]                                                 |

