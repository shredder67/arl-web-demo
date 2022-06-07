import pandas as pd

def aggregate_transactions(df : pd.DataFrame):
    assert len(df.columns) == 2, 'Transactions dataframe wrong format'

    user_ids = df.iloc[:, 0].unique().tolist()
    items = df.iloc[:, 1].unique().tolist()
    new_df = pd.DataFrame(index=user_ids, columns=items)
    for idx, tr in df.iterrows():
        uid, title = tr.iloc[0], tr.iloc[1]
        new_df[title][uid] = 1
    new_df = new_df.fillna(0)
    return new_df