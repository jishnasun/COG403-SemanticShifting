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
| `language_ranking_full.csv`                             | Languages ranked by accuracy of `testBayesianLanguage.py`.                      |
| `language_ranking_glotto.csv`  | Languages ranked by accuracy of `testBayesianGlotto.py`.                                                 |
<br>

Relevant code regarding models, evaluation, and visualization of results:

| File                                  | Description                                                                                                                        |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `testBayesianLanguage.py`                             | [description]                      |
| `testBayesianGlotto.py`  | [description]                                                 |
| `test_file_2.0.py`  | [description]                                                 |
| `train_test_2.0.py`  | [description]                                                 |
| `test_file_2.0.py`  | [description]                                                 |
| `singLangTestCase.py`  | [description]                                                 |
| `testCaseGlotto.py`  | [description]                                                 |
| `results.py`  | [description]                                                 |

