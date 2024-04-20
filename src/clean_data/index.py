import pandas as pd

from src.explore_data.index import miss_data_graph
from src.util.CleanData import CleanData
from src.util.DataNormalization import DataNormalization


def clean(df: pd.DataFrame):
    df.rename(columns=CleanData.clean_col)
    miss_data_graph(df)
    titanic_clean = DataNormalization.tweak_titanic(df)
    return CleanData.input_empty_values(titanic_clean)