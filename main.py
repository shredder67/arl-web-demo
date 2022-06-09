import streamlit as st
import pandas as pd
import numpy as np

from source.arl.utils import aggregate_transactions, hash_dataframe
from source.arl.ARL import MyARL

class DataState:
    def __init__(self, data : pd.DataFrame, min_sup, min_conf):
        self.data = data
        self.min_sup = min_sup
        self.min_conf = min_conf

    def __hash__(self):
        return hash_dataframe(self.data) ^\
            hash(self.min_sup) ^\
            hash(self.min_conf)

            

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

        if type(source_data) is not pd.DataFrame:
            source_data = pd.read_csv(source_data)

        min_support = st.sidebar.slider(
            'Минимальная поддержка:',
            min_value=0.05,
            max_value=1.0,
            value=0.15
        )

        min_confidence = st.sidebar.slider(
            'Минимальная достоверность:',
            min_value=0.05,
            max_value=1.0,
            value=0.6
        )

        display_mode = st.sidebar.selectbox(
        'Способ представления данных',
        ['Популярные наборы', 'Правила']
        )

        'Исходные данные:', source_data

        aggr_df = aggregate_transactions(source_data)

        arl = None
        ds = DataState(aggr_df, min_support, min_confidence)
        if 'ds_hash' not in st.session_state:
            st.session_state['ds_hash'] = hash(ds)
        elif hash(ds) == st.session_state.ds_hash:
            arl = st.session_state.arl
        
        if arl is None:
            arl = MyARL()
            arl.apriori(aggr_df.values, min_support=min_support, min_confidence=min_confidence, labels=aggr_df.columns)
            st.session_state['arl'] = arl

        arl_rules = arl.get_rules()
        arl_pop_itemssets = arl.get_popular_itemsets()

        if display_mode == 'Популярные наборы':
            arl_pop_itemssets = arl.get_popular_itemsets()
            if len(arl_pop_itemssets) > 0:
                itemsets, it_supports = zip(*arl_pop_itemssets)
                popular_itemsts_df = pd.DataFrame(data={
                    'Itemset': [", ".join(it_set) for it_set in itemsets],
                    'Support': it_supports
                })
                popular_itemsts_df.index += 1
                

                'Популярные наборы:'
                st.write(popular_itemsts_df.style.format(
                    subset=["Support"], 
                    formatter='{:.2f}'
                ))
            else:
                'Для заданного параметра поддержки не было найдено популярных наборов!'
        elif display_mode == 'Правила':
            arl_rules = arl.get_rules()
            if len(arl_rules) > 0:
                antecedents, consequents, supports, confidences, lifts = zip(*arl_rules)
                rules_df = pd.DataFrame(data={
                    "Antecedent": [", ".join(it_set) for it_set in antecedents],
                    "Consequent": [", ".join(it_set) for it_set in consequents],
                    "Support": supports,
                    "Confidence": confidences,
                    "Lift": lifts
                })
                rules_df.index += 1


                'Правила:'
                st.write(rules_df.style.format(
                    subset=["Support", "Confidence", "Lift"],
                    formatter='{:.2f}'
                ))
            else:
                'Для заданных параметров не было найдено популярных правил!'


if __name__ == '__main__':
    main()