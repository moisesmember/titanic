import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from src.data_preprocessing.class_balancing.index import class_balance
from src.data_preprocessing.columns.index import correlated_columns, lasso_graph, recursive_attribute_deletion
from src.ml.classification.classification import classification
from src.util.BaseModelAnalysis import BaseModelAnalysis
from src.util.DataNormalization import DataNormalization


def preprocessing(df: pd.DataFrame):
    correlated_columns_df = correlated_columns(df)
    correlated_columns_df.style.format({"pearson": "{:.2f"})

    std_cols = "pclass,age,sibsp,fare".split(",")
    X_train, X_test, y_train, y_test = DataNormalization.get_train_test_X_y(
        df, "survived", std_cols=std_cols
    )
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    BaseModelAnalysis.analysis(X, y)

    lasso_graph(X, X_train, y_train) # Seleção de atributos

    model = ensemble.RandomForestClassifier(n_estimators=100)
    columns_selected = recursive_attribute_deletion(X, y, model)

    X_smote, y_smote = class_balance(X, y)

    new_data = pd.concat([X_smote[columns_selected], y_smote], axis=1)

    print(df.survived.value_counts())
    print(new_data.survived.value_counts())

    classification(new_data, columns_selected)

