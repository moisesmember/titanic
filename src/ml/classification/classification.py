from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.util.DataNormalization import DataNormalization


def classification(titanic_df: pd.DataFrame, columns: List):
    columns_formatted = list(titanic_df[columns].select_dtypes(include=['number']).columns)
    X_train, X_test, y_train, y_test = DataNormalization.get_train_test_X_y(
        titanic_df, "survived", std_cols=columns_formatted
    )

    print("++++++++++++++++++++++++++++++++++++++++++++++++++")

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    print(lr.score(X_test, y_test))

    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    print(lr.predict(X.iloc[[0]]))
