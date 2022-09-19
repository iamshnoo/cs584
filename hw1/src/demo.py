import pandas as pd

df["target"] = df["target"].apply(lambda x: "+1" if x == 1 else "-1")
