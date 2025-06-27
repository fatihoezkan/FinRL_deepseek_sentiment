import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Nvidia Stock Predictions", layout="wide")

st.markdown(
    "<h1 style='text-align: center; font-weight: bold;'>Nvidia Stock (near) real-time predictions with FinRL and DeepSeek</h1>",
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    df = pd.read_csv('../results.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').reset_index()
    df['A2C % Diff'] = 100 * (df["A2C Agent 2"] - df["A2C Agent 1"]) / df["A2C Agent 1"]
    df['SAC % Diff'] = 100 * (df["SAC Agent 2"] - df["SAC Agent 1"]) / df["SAC Agent 1"]
    df['PPO % Diff'] = 100 * (df["PPO Agent 2"] - df["PPO Agent 1"]) / df["PPO Agent 1"]
    df['TD3 % Diff'] = 100 * (df["TD3 Agent 2"] - df["TD3 Agent 1"]) / df["TD3 Agent 1"]
    df['A2C % Diff 3'] = 100 * (df["A2C Agent 3"] - df["A2C Agent 1"]) / df["A2C Agent 1"]
    df['SAC % Diff 3'] = 100 * (df["SAC Agent 3"] - df["SAC Agent 1"]) / df["SAC Agent 1"]
    df['PPO % Diff 3'] = 100 * (df["PPO Agent 3"] - df["PPO Agent 1"]) / df["PPO Agent 1"]
    df['TD3 % Diff 3'] = 100 * (df["TD3 Agent 3"] - df["TD3 Agent 1"]) / df["TD3 Agent 1"]
    return df

df = load_data()

# Include all diff columns in id_vars
id_vars = [
    'date', 'A2C % Diff', 'SAC % Diff', 'PPO % Diff', 'TD3 % Diff',
    'A2C % Diff 3', 'SAC % Diff 3', 'PPO % Diff 3', 'TD3 % Diff 3'
]
df_melted = df.melt(
    id_vars=id_vars, 
    var_name='Agent', 
    value_name='Portfolio Value'
)

custom_colors = {
    "A2C Agent 1": "#CD5C5C",
    "A2C Agent 2": "#FF8C00",
    "A2C Agent 3": "#FF4500",
    "SAC Agent 1": "#4169E1",
    "SAC Agent 2": "#9370DB",
    "SAC Agent 3": "#00CED1",
    "PPO Agent 1": "#32CD32",
    "PPO Agent 2": "#FFD700",
    "PPO Agent 3": "#FF6347",
    "TD3 Agent 1": "#FF69B4",
    "TD3 Agent 2": "#8A2BE2",
    "TD3 Agent 3": "#7FFF00",
    "Mean Var": "#5F9EA0",
    "djia": "#9ACD32"
}

def compute_tooltip(row):
    try:
        if row['Agent'] == 'A2C Agent 2':
            return f"{row['A2C % Diff']:.2f}%"
        elif row['Agent'] == 'SAC Agent 2':
            return f"{row['SAC % Diff']:.2f}%"
        elif row['Agent'] == 'PPO Agent 2':
            return f"{row['PPO % Diff']:.2f}%"
        elif row['Agent'] == 'TD3 Agent 2':
            return f"{row['TD3 % Diff']:.2f}%"
        elif row['Agent'] == 'A2C Agent 3':
            return f"{row['A2C % Diff 3']:.2f}%"
        elif row['Agent'] == 'SAC Agent 3':
            return f"{row['SAC % Diff 3']:.2f}%"
        elif row['Agent'] == 'PPO Agent 3':
            return f"{row['PPO % Diff 3']:.2f}%"
        elif row['Agent'] == 'TD3 Agent 3':
            return f"{row['TD3 % Diff 3']:.2f}%"
        else:
            return "-"
    except Exception:
        return "-"

df_melted['Tooltip Diff'] = df_melted.apply(compute_tooltip, axis=1)

chart = alt.Chart(df_melted).mark_line().encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('Portfolio Value:Q', title='Portfolio Value'),
    color=alt.Color('Agent:N', scale=alt.Scale(domain=list(custom_colors.keys()), range=list(custom_colors.values()))),
    tooltip=[
        alt.Tooltip('date:T', title='Date'),
        alt.Tooltip('Agent:N', title='Agent'),
        alt.Tooltip('Portfolio Value:Q', title='Value', format=".2f"),
        alt.Tooltip('Tooltip Diff:N', title='% Diff vs Agent 1')
    ]
).interactive().properties(
    width=1000,
    height=500,
    title='Portfolio Performance Over Time'
)

st.altair_chart(chart, use_container_width=True)