import pandas as pd
from sklearn import (
    preprocessing,
)
from sklearn.model_selection import (
    train_test_split
)
from sklearn.experimental import enable_iterative_imputer


class DataNormalization:
    @staticmethod
    def tweak_titanic(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(
            columns=[
                "name", "ticket", "home.dest", "boat", "body", "cabin"
            ]
        ).pipe(pd.get_dummies, drop_first=True)
        return df

    @staticmethod
    def get_train_test_X_y(
            df: pd.DataFrame, y_col, size=0.3, std_cols=None
    ):
        y = df[y_col]
        X = df.drop(columns=y_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
        num_cols = X.select_dtypes(include=['number']).columns
        fi = enable_iterative_imputer.IterativeImputer()
        X_train.loc[:, num_cols] = fi.fit_transform(X_train[num_cols])
        X_test.loc[:, num_cols] = fi.fit_transform(X_test[num_cols])

        if std_cols:
            print(std_cols)
            std = preprocessing.StandardScaler()
            assert X_train[std_cols].select_dtypes(include='number').equals(
                X_train[std_cols]), "Not all columns in std_cols are numeric"
            X_train.loc[:, std_cols] = std.fit_transform(X_train[std_cols])
            X_test.loc[:, std_cols] = std.transform(X_test[std_cols])

        return X_train, X_test, y_train, y_test