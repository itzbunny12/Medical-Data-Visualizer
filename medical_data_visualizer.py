import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Import the data
df = pd.read_csv('medical_examination.csv')

# 2: Add overweight column (BMI > 25)
height_m = df['height'] / 100
bmi = df['weight'] / (height_m ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3: Normalize 'cholesterol' and 'gluc' columns: 0 if normal (1), else 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4: Draw Categorical Plot
def draw_cat_plot():
    # 5: Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6: Group and reformat the data to get counts
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7: Draw the categorical plot with seaborn
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio',
                      data=df_cat, kind='bar').fig

    # 9: Save and return figure
    fig.savefig('catplot.png')
    return fig


# 10: Draw Heat Map
def draw_heat_map():
    # 11: Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12: Calculate correlation matrix
    corr = df_heat.corr()

    # 13: Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15: Draw heatmap with seaborn
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, square=True,
                linewidths=.5, ax=ax, center=0, cmap='coolwarm')

    # 16: Save and return figure
    fig.savefig('heatmap.png')
    return fig
