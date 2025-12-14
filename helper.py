from tqdm import tqdm
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import combinations

def count_nulls(df: pd.DataFrame) -> pd.DataFrame:
    # return data in format, col : count of nulls, percent missing of total
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    return pd.DataFrame({'nulls': nulls, 'percent_missing': nulls / df.shape[0]})


def ajr_plot_correlations(df: pd.DataFrame, y_comp_name: str, fig_height=20, save_to = None,r_squared=False) -> None:
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        print(f'Dropping {null_counts}/{df.shape[0]} rows with null values')
        df = df.dropna()
    # ajr plotting correlations
    n_plots = len(df.columns) - 1
    n_rows = 1 if n_plots <= 4 else 2 if n_plots <= 8 else 3 if n_plots <= 12 else 4 if n_plots <= 16 else 5 if n_plots <= 20 else 6
    n_cols = int(np.ceil(n_plots / n_rows))
    fig_width = fig_height * n_rows / n_cols
    marker_size = fig_height - fig_height / 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_height, fig_width))
    ax = ax.flatten()
    for i, col in enumerate(df.columns.drop(y_comp_name)):
        lm = LinearRegression()
        x = df[col].values.reshape(-1, 1)
        y = df[y_comp_name]
        lm.fit(x, y)
        ax[i].scatter(x, y, color='b', s=marker_size, alpha=0.5)
        ax[i].plot(x, lm.predict(x), color="r")

        title = f'{col} | Corr: {round(df[col].corr(df[y_comp_name]), 2)}'
        title += f' | R^2: {round(lm.score(x, y), 2)}' if r_squared else ''
        ax[i].set_title(title)
        ax[i].set_xlabel(col)
        ax[i].set_ylabel(y_comp_name)
    for j in range(i + 1, len(ax)):
        ax[j].axis('off')
    plt.tight_layout()
    if save_to:
        fig.savefig(save_to)
        print(f'Figure saved to {save_to}')
    plt.show()
    
def ajr_correlation_heatmap(corr_df: pd.DataFrame, fig_height=10) -> None:
    column_names = list(corr_df.columns.values)
    plt.figure(figsize=(fig_height, fig_height))
    plt.matshow(corr_df, cmap='coolwarm', fignum=1)
    plt.xticks(np.arange(len(column_names)), column_names, rotation=90)
    plt.yticks(np.arange(len(column_names)), column_names)
    plt.tight_layout()
    for i in range(len(corr_df.columns)):
        for j in range(len(corr_df.columns)):
            plt.text(j, i, f'{round(corr_df.iloc[i, j], 2)}', ha='center', va='center', color='black')
    plt.show()
    
    
def ajr_plot_histograms(df: pd.DataFrame, save_to = None) -> None:
    a = int(np.ceil(np.sqrt(df.shape[1])))
    b = int(np.ceil(df.shape[1] / a))
    fig, ax = plt.subplots(a, b, figsize=(20, 20))
    for i, col in enumerate(df.columns):
        c = df[col]
        ax[i // b, i % b].hist(c, bins=20)
        title = f'{col} (na: {c.isnull().sum() / c.shape[0] * 100:.2f}%)'
        ax[i // b, i % b].set_title(title)
    plt.tight_layout()
    if save_to:
        fig.savefig(save_to)
        print(f'Figure saved to {save_to}')
    plt.show()


def ajr_plot_scatter_volume(df: pd.DataFrame, x: str, y: str, fig_height=12):
    fig, (ax_1, ax_2) = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]}, figsize=(fig_height, fig_height/2))

    ax_1.scatter(df[x], df[y], alpha=0.5)
    ax_1.set_xlabel(x)
    ax_1.set_ylabel(y)
    ax_1.set_title(f'{x} vs. {y}')

    ax_2.hist(df[x], bins=50, alpha=0.5, label=x)
    #ax_2.hist(df[y], bins=50, alpha=0.5, label=y)
    ax_2.set_xlabel('Value')
    ax_2.set_ylabel('Frequency')
    #ax_2.legend()
    ax_2.grid()
    plt.show()
    

def ajr_test_models(X, y, models):
    accuracies = {}
    for model in models:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[model] = r2_score(y_test, y_pred)
    return dict(sorted(accuracies.items(), key=lambda x: x[1], reverse=True))


def best_model_accuracy(accuracies: dict) -> tuple:
    best_model = None
    best_accuracy = 0
    for model, accuracy in accuracies.items():
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy
    return best_model, best_accuracy


def combs(features: list) -> list:
    feature_combs = []
    for r in range(1, len(features) + 1):
        feature_combs += list(combinations(features, r))
    return feature_combs



def ajr_find_best_combination(features: list, target: str, df: pd.DataFrame, models: list) -> tuple:
    combinations = combs(features)
    combination_scores = {}
    
    for comb in tqdm(combinations):
        X = df[list(comb)].values
        y = df[target].values
        accuracies = ajr_test_models(X, y, models)
        best_model, best_accuracy = best_model_accuracy(accuracies)
        combination_scores[comb] = (best_model, best_accuracy)
        
    best_combination = max(combination_scores, key=lambda x: combination_scores[x][1])
    best_model, best_accuracy = combination_scores[best_combination]
    return best_combination, best_model, best_accuracy

