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
        uid, title = tr[0], tr[1]
        new_df[title][uid] = 1
    new_df = new_df.fillna(0)
    return new_df


def hash_dataframe(df):
    """
    Hashes whole dataframe object

    Parameters:
        df (pandas.DataFrame) - target dataframe
    
    Return:
        res_hash (int) - hashing result
    """
    row_hashes = pd.util.hash_pandas_object(df).to_list()
    res_hash = 0
    for h in row_hashes:
        res_hash ^= h
    return res_hash