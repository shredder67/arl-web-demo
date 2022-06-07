import streamlit as st
import pandas as pd
import numpy as np

from source.arl.data_transform import aggregate_transactions
from source.arl.ARL import MyARL

selected_file = None

def main():
    source_data = None

    if st.sidebar.checkbox('Показать на примере'):
        source_data = pd.read_csv('./source/data/example_input.csv')

    if source_data is None:
        source_data = st.sidebar.file_uploader(
            'Загрузите файл с транзакциями',
            type='csv',
            accept_multiple_files=False
        )
        
    if source_data is not None:
        'Исходные данные:', source_data

        aggr_df = aggregate_transactions(source_data)

        arl = MyARL()
        arl.apriori(aggr_df.values, min_support=0.15, min_confidence=0.6, labels=aggr_df.columns)

        antecedents, consequents, supports, confidences, lifts = zip(*arl.get_rules())
        itemsets, it_supports = zip(*arl.get_popular_itemsets())

        rules_df = pd.DataFrame(data={
            "Antecedent": antecedents,
            "Consequent": consequents,
            "Support": supports,
            "Confidence": confidences,
            "Lift": lifts
        })
        rules_df.index += 1
        popular_itemsts_df = pd.DataFrame(data={
        'Itemset': itemsets,
        'Support': it_supports
        })
        popular_itemsts_df.index += 1
        

        displayed_model = st.sidebar.selectbox(
        'Способ представления данных',
        ['Популярные наборы', 'Правила']
        )

        if displayed_model == 'Популярные наборы':
            'Найденные правила:', rules_df
        elif displayed_model == 'Правила':
            'Найденные популярные наборы:', popular_itemsts_df
    



if __name__ == '__main__':
    main()