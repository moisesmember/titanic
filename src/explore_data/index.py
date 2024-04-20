import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from seaborn import pairplot, boxplot, heatmap
from yellowbrick.features import (
    JointPlotVisualizer, RadViz
)
from src.util.DataNormalization import DataNormalization


def explore(df: pd.DataFrame):
    std_cols = "pclass,age,sibsp,fare".split(",")
    X_train, X_test, y_train, y_test = DataNormalization.get_train_test_X_y(
        df, "survived", std_cols=std_cols
    )
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    # Distribuição dos preços
    fig, ax = plt.subplots(figsize=(6, 4))
    df['fare'].plot(kind="hist", ax=ax)
    fig.savefig("./assets/explore_data/fare-distribution.png", dpi=300)

    # Distribuição dos alvos (sobrevivente ou não)
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = y_train == 1
    ax = sns.histplot(X_train[mask]['fare'], label='survived')
    ax = sns.histplot(X_train[~mask]['fare'], label='died')
    ax.set_xlim(-1.5, 1.5)
    ax.legend()
    fig.savefig("./assets/explore_data/target-distribution.png", dpi=300, bbox_inches='tight')

    # Relacionamento entre idade e preços
    fig, ax = plt.subplots(figsize=(6, 6))
    jpv = JointPlotVisualizer(
        feature="age", target="fare"
    )
    jpv.fit(X["age"], X["fare"])
    #jpv.poof()
    fig.savefig("./assets/explore_data/age-fare-relationship.png", dpi=300)

    fig, ax = plt.subplots(figsize=(14, 14))
    new_df = X.copy()
    new_df["target"] = y
    vars = ["pclass", "age", "fare"]
    graph_pairplot = pairplot(
        new_df, vars=vars, hue="target", kind="reg"
    )
    graph_pairplot.savefig("./assets/explore_data/pair-matrix.png", dpi=300)

    fig, ax = plt.subplots(figsize=(16, 14))
    new_df = X.copy()
    new_df["target"] = y
    boxplot(x="target", y="age", data=new_df)
    fig.savefig("./assets/explore_data/boxplot-age.png", dpi=300)

    fig, ax = plt.subplots(figsize=(16, 14))
    ax = heatmap(
        X.corr(),
        fmt=".2f",
        annot=True,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    fig.savefig("./assets/explore_data/heatmap-attribute-correlation.png", dpi=300, bbox_inches="tight")

    # fig, ax = plt.subplots(figsize=(16, 14))
    # rv = RadViz(
    #     classes=["died", "survived"],
    #     features=X.columns,
    # )
    # rv.fit(X, y)
    # _ = rv.transform(X)
    # fig.savefig("./assets/explore_data/degree-separation-targets.png", dpi=300)

def miss_data_graph(df: pd.DataFrame):
    ax = msno.matrix(df)
    ax.get_figure().savefig("./assets/clean_data/miss_values.png")

    dendrogram = msno.dendrogram(df)
    dendrogram.get_figure().savefig("./assets/clean_data/clustering_values.png")