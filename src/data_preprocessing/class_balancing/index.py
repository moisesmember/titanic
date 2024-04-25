import numpy as np
import pandas as pd
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE
)
from sklearn.utils import resample


def class_balance(X: pd.DataFrame, y: pd.DataFrame):
    print("\n\n************ BALANCIAMENTO DAS CLASSES ************")
    # mask = df['survived'] == 1
    # surv_df = df[mask]
    # death_df = df[~mask]
    # df_upsample = resample(
    #     surv_df,
    #     replace=True,
    #     n_samples=len(death_df),
    #     random_state=42,
    # )
    # df2 = pd.concat([death_df, df_upsample])
    # print(df2.survived.value_counts())

    # RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X, y)

    # SMOTE (opcional, se desejar usar)
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Exibir as contagens das classes após a reamostragem
    print("\nContagem das classes após a aplicação do RandomOverSampler:")
    print(pd.Series(y_ros).value_counts())

    # Se desejar, exibir as contagens das classes após a aplicação do SMOTE
    print("\nContagem das classes após a aplicação do SMOTE:")
    print(pd.Series(y_smote).value_counts())

    return X_smote, y_smote