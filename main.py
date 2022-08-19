import streamlit as st
import pandas as pd

from source.arl.utils import transform_df_to_item_counts
from source.arl.ARL import MyARL, format_rules_into_df, format_pop_itemsets_into_df


APPLY_CSS = False


def check_df_format(df: pd.DataFrame) -> bool:
    '''
    Checks dataframe for accordance to defined transactions format.

    Format of dataframe columns should be like {id: int, : any}

    ### Parameters:
    df: pd.DataFrame - target dataframe

    ### Returns:
    isValid: bool - is dataframe correct or not
    '''
    return df.shape[1] == 2 and df.dtypes[0] == 'int64' and df.columns[0] == 'id'


def apply_css_style(style_source: str) -> None:
    '''
    Applies css styles to markdown through streamlit API. May break everything, use with caution

    ### Parameters:

    style_source: str - path to .css file
    '''
    with open(style_source) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def main() -> None:
    '''
    App entry point and loop, runs after every page reload/interaction
    '''
    if APPLY_CSS:
        apply_css_style('source/public/style.css')

    source_data = None
    source_is_correct = True

    if st.sidebar.checkbox('Показать на примере'):
        source_data = './source/data/data_example.csv'
    elif source_data is None:
        source_data = st.sidebar.file_uploader(
            'Загрузите файл csv с транзакциями',
            type='csv',
            accept_multiple_files=False,
            help='Данны должны быть структурированы в две колонки, у первой название должно быть id'
        )

    if not source_data: return

    source_data = pd.read_csv(source_data)
    source_is_correct = check_df_format(source_data)

    if not source_is_correct:
        st.warning('Неправильный формат загруженных данных!')
        st.write(source_data.sample(1))
        return

    source_col, output_col = st.columns(2, gap='large')
    
    with st.sidebar.form(key='rule_params_form'):
        min_support = st.slider(
            'Минимальная поддержка:',
            min_value=0.05,
            max_value=1.0,
            value=0.15
        )
        min_confidence = st.slider(
            'Минимальная достоверность:',
            min_value=0.05,
            max_value=1.0,
            value=0.6
        )
        display_mode = st.selectbox(
        'Способ представления данных',
        ['Популярные наборы', 'Правила', 'Что-Если']
        )
        st.form_submit_button("Выполнить поиск")
    
    with source_col:
        'Исходные данные:', source_data.groupby(['id']).aggregate(func=lambda vals: ', '.join(vals))

    aggr_df = transform_df_to_item_counts(source_data)

    arl = MyARL()
    arl.apriori(aggr_df.values, min_support, min_confidence, labels=aggr_df.columns)

    arl_rules = arl.get_rules()
    arl_pop_itemsets = arl.get_popular_itemsets()
    with output_col:
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