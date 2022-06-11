import streamlit as st
import pandas as pd
import numpy as np

from source.arl.utils import hash_dataframe, transform_df_to_item_counts
from source.arl.ARL import MyARL, format_rules_into_df, format_pop_itemsets_into_df

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
        source_data = pd.read_csv('./source/data/data_example.csv')

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
        ['Популярные наборы', 'Правила', 'Что-Если']
        )

        'Исходные данные:', source_data.groupby(['id']).aggregate(func=lambda vals: ', '.join(vals))

        aggr_df = transform_df_to_item_counts(source_data)

        arl = None
        ds = DataState(aggr_df, min_support, min_confidence)
        if 'ds_hash' not in st.session_state:
            st.session_state['ds_hash'] = hash(ds)
        elif hash(ds) == st.session_state.ds_hash:
            arl = st.session_state.arl

        if arl is None:
            arl = MyARL()
            arl.apriori(aggr_df.values, min_support, min_confidence, labels=aggr_df.columns)
            st.session_state['arl'] = arl

        arl_rules = arl.get_rules()
        arl_pop_itemsets = arl.get_popular_itemsets()

        if display_mode == 'Популярные наборы':
            if len(arl_pop_itemsets) > 0:
                popular_itemsets_df = format_pop_itemsets_into_df(arl_pop_itemsets)

                'Популярные наборы:'
                st.write(popular_itemsets_df.style.format(
                    subset=["Support"], 
                    formatter='{:.2f}'
                ))
            else:
                'Для заданного параметра поддержки не было найдено популярных наборов!'
        elif display_mode == 'Правила':
            if len(arl_rules) > 0:
                rules_df = format_rules_into_df(arl_rules)
                'Правила:'
                st.write(rules_df.style.format(
                    subset=["Support", "Confidence", "Lift"],
                    formatter='{:.2f}'
                ))
            else:
                'Для заданных параметров не было найдено популярных правил!'
        elif display_mode == 'Что-Если':
            if len(arl_rules) > 0:
                rules_df = format_rules_into_df(arl_rules)
                
                selected_ant = st.multiselect(
                    'Если:',
                    options=aggr_df.columns,
                    default=None
                )

                conseq_df = pd.DataFrame(columns=['Consequent', 'Support', 'Confidence', 'Lift'])
                for ant in selected_ant:
                    for idx, rule in rules_df.iterrows():
                        rule_ant = rule[0]
                        if len(rule_ant) == 1 and ant == rule_ant[0]:
                            conseq_df = conseq_df.append(
                                rule[1:],
                                ignore_index=True,
                        )
                if len(conseq_df) > 0:
                    'Что:'
                    st.write(conseq_df.style.format(
                        subset=["Support", "Confidence", "Lift"],
                        formatter='{:.2f}'
                    ))
                else:
                    'Для выбранных предметов не найдены подходящие следствия!'
            


if __name__ == '__main__':
    main()