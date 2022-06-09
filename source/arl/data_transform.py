import pandas as pd

def aggregate_transactions(df : pd.DataFrame):
    """
    Transforms pandas dataframe with form like {id, item} into
    {id, [item_cols]}, where one row includes counts of all
    items related to one id (which are now unique)

    Parameters:
        df (pandas.DataFrame): source dataframe with transactions
    
    Returns:
        new_df (pandas.DataFrame): formatted dataframe
     """
    assert len(df.columns) == 2, 'Transactions dataframe wrong format'

    user_ids = df.iloc[:, 0].unique().tolist()
    items = df.iloc[:, 1].unique().tolist()
    new_df = pd.DataFrame(index=user_ids, columns=items)
    for idx, tr in df.iterrows():
        uid, title = tr.iloc[0], tr.iloc[1]
        new_df[title][uid] = 1
    new_df = new_df.fillna(0)
    return new_df