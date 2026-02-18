import pandas as pd

pathb = "COG403-SemanticShifting/cldf/"  # Change path to cldf directory
values = pd.read_csv(pathb + "ValueTable.csv")
forms = pd.read_csv(pathb + "FormTable.csv")
languages = pd.read_csv(pathb + "LanguageTable.csv")
