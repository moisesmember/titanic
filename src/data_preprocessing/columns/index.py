import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from yellowbrick.model_selection import RFECV


def correlated_columns(df: pd.DataFrame, threshold=0.95):
    return (
        df.corr()
        .pipe(
            lambda df1: pd.DataFrame(
                np.tril(df1, k=-1),
                columns=df.columns,
                index=df.columns,
            )
        )
        .stack()
        .rename("pearson")
        .pipe(
            lambda s: s[
                s.abs() > threshold
            ].reset_index()
        )
        .query("level_0 not in level_1")
    )


def lasso_graph(X: pd.DataFrame, X_train, y_train):
    model = linear_model.LassoLarsCV(
        cv=10, max_n_alphas=10
    ).fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(12, 8))
    cm = iter(
        plt.get_cmap("tab20")(
            np.linspace(0, 1, X.shape[1])
        )
    )
    for i in range(X.shape[1]):
        c = next(cm)
        ax.plot(
            model.alphas_,
            model.coef_path_.T[:, i],
            c=c,
            alpha=0.8,
            label=X.columns[i],
        )
    ax.axvline(
        model.alpha_,
        linestyle="-",
        c="k",
        label="alphaCV",
    )
    plt.ylabel("Coeficiente de Regressão")
    ax.legend(X.columns, bbox_to_anchor=(1, 1))
    plt.xlabel("alpha")
    plt.title("Pogressão do Coeficiente de regressão Lasso")
    fig.savefig("./assets/data_preprocessing/columns/regression-lasso-columns.png", dpi=300, bbox_inches='tight')


# Eliminação Recursiva de Atributos com Validação Cruzada
def recursive_attribute_deletion(X: pd.DataFrame, y: pd.DataFrame, model):
    fig, ax = plt.subplots(figsize=(12, 8))
    rfe = RFECV(model, cv=5)
    rfe.fit(X, y)

    print("\n\n************ Eliminação Recursiva de Atributos com Validação Cruzada ************")

    # Ranking dos atributos. 1º é atributo mai importante, 2º é o segundo mais importante.
    print("Ranking dos atributos: ", rfe.rfe_estimator_.ranking_)

    # Número ideal de atributos para esse modelo
    print("Número ideal de atributos: ", rfe.rfe_estimator_.n_features_)

    #  Este atributo é um array booleano que indica quais atributos foram selecionados. ESCOLHIDO [True], ELIMINADO [False]
    # print("Atributos selecionados[TRUE] e eliminados[False]: \n", rfe.rfe_estimator_.support_)
    print("Atributos eliminados: ", [column for column, condition in zip(X.columns, rfe.rfe_estimator_.support_) if not condition])

    plt.ylabel("Score")
    ax.legend(["Score"], bbox_to_anchor=(1, 1))
    plt.xlabel("Nº de atributos selecionados")
    plt.title("Eliminação Recursiva de Atributos com Validação Cruzada")
    fig.savefig("./assets/data_preprocessing/columns/recursive_attribute_deletion.png", dpi=300)

    columns_selected = list(X.columns[rfe.rfe_estimator_.support_])
    return columns_selected
