import marimo

__generated_with = "0.17.7"
app = marimo.App(
    width="medium",
    layout_file="layouts/ai_research_demo.slides.json",
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Linear Regression Example

    ## ... or can I just ask AI to "analyze the data"?

    Imagine that you've asked a chatbot how to run linear regression on four sets of data.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from scipy import stats

    np.random.seed(42)
    sns.set_style("whitegrid")
    return LinearRegression, pd, plt, r2_score, stats


@app.cell
def _(mo):
    mo.md("""
    ### Anscombe's Quartet

    Let's see what happens.
    """)
    return


@app.cell
def _(pd):
    anscombe = pd.DataFrame({
        'x1': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y1': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        'x2': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y2': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        'x3': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y3': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
        'x4': [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
        'y4': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
    })
    return (anscombe,)


@app.cell
def _(anscombe):
    anscombe
    return


@app.cell
def _(LinearRegression, anscombe, pd, r2_score):
    results = []
    for i in range(1, 5):
        X = anscombe[f'x{i}'].values.reshape(-1, 1)
        y = anscombe[f'y{i}'].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        slope = model.coef_[0]
        intercept = model.intercept_

        results.append({
            'Dataset': i,
            'R²': f'{r2:.3f}',
            'Slope': f'{slope:.2f}',
            'Intercept': f'{intercept:.2f}'
        })

    results_df = pd.DataFrame(results)
    results_df
    return


@app.cell
def _(mo):
    mo.md("""
    #### Incredible! (in the most literal sense) we're getting identical $R^2$ and identical lines for 4 different sets of data.

    ### But let's plot it...
    """)
    return


@app.cell
def _(LinearRegression, anscombe, plt):
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for j in range(1, 5):
        ax = axes[j-1]
        X_plot = anscombe[f'x{j}'].values.reshape(-1, 1)
        y_plot = anscombe[f'y{j}'].values

        # Fit and plot
        model_plot = LinearRegression()
        model_plot.fit(X_plot, y_plot)
        y_pred_plot = model_plot.predict(X_plot)

        # Scatter
        ax.scatter(X_plot, y_plot, s=100, alpha=0.6)
        ax.plot(X_plot, y_pred_plot, 'r-', linewidth=2, label='Linear Fit')

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'Dataset {j}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim(3, 13)

    plt.tight_layout()
    fig1
    return


@app.cell
def _(mo):
    mo.md("""
    ### Well, that's not great.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Example 2: Auto MPG

    Relationship between engine horsepower and fuel efficiency (MPG).
    """)
    return


@app.cell
def _(pd):
    # Load auto mpg dataset
    url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
    mpg_data = pd.read_csv(url)

    # Clean data
    mpg_clean = mpg_data[['horsepower', 'mpg']].dropna()
    mpg_clean
    return (mpg_clean,)


@app.cell
def _(LinearRegression, mpg_clean, r2_score):
    X_mpg = mpg_clean['horsepower'].values.reshape(-1, 1)
    y_mpg = mpg_clean['mpg'].values

    model_mpg = LinearRegression()
    model_mpg.fit(X_mpg, y_mpg)
    y_pred_mpg = model_mpg.predict(X_mpg)
    residuals_mpg = y_mpg - y_pred_mpg

    r2_mpg = r2_score(y_mpg, y_pred_mpg)

    print(f"R² = {r2_mpg:.3f}")
    print(f"Slope: {model_mpg.coef_[0]}")
    return X_mpg, model_mpg, r2_mpg, residuals_mpg, y_mpg, y_pred_mpg


@app.cell
def _(mo, model_mpg, r2_mpg):
    mo.md(f"""
    ### $R^2$ = {r2_mpg:.3f} and **Slope** = {model_mpg.coef_[0]: .3f}

    Great! Negatively correlated, as expected, with reasonably high explainability.

    But there's a problem when we look at the graph.
    """)
    return


@app.cell
def _(X_mpg, plt, residuals_mpg, stats, y_mpg, y_pred_mpg):
    # Show what you'd miss without checking
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    # Raw data with fit
    axes2[0].scatter(X_mpg, y_mpg, alpha=0.5, s=20)
    axes2[0].plot(X_mpg, y_pred_mpg, 'r-', linewidth=2)
    axes2[0].set_xlabel('Horsepower', fontsize=12)
    axes2[0].set_ylabel('MPG', fontsize=12)
    axes2[0].set_title('Linear Fit', fontsize=14, fontweight='bold')

    # Residual plot
    axes2[1].scatter(y_pred_mpg, residuals_mpg, alpha=0.5, s=20)
    axes2[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes2[1].set_xlabel('Fitted Values', fontsize=12)
    axes2[1].set_ylabel('Residuals', fontsize=12)
    axes2[1].set_title('Residuals', fontsize=12, fontweight = 'bold')

    # Q-Q plot
    stats.probplot(residuals_mpg, dist="norm", plot=axes2[2])
    axes2[2].set_title('Q-Q Plot', fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig2
    return


@app.cell
def _(mo, r2_mpg):
    mo.md(f"""
    ### The Problem:

    $R^2$ = {r2_mpg:.3f} looks decent, but:
    - The plots shows a clear nonlinearity in the data.
    - Residuals are not normally distributed

    The linear model is wrong.

    **Just asking an LLM for a linear regression, you might not catch this unless you tell it to check.**
    """)
    return


if __name__ == "__main__":
    app.run()
