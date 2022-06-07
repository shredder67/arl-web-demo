import pandas as pd
from ARL import MyARL

def main():
    
    data = [(0, 'Капуста'),
(0, 'Перец'),
(0, 'Кукуруза'),
(1, 'Спаржа'),
(1, 'Кабачки'),
(1, 'Кукуруза'),
(2, 'Кукуруза'),
(2, 'Помидоры'),
(2, 'Фасоль'),
(2, 'Кабачки'),
(3, 'Перец'),
(3, 'Кукуруза'),
(3, 'Помидоры'),
(3, 'Фасоль'),
(4, 'Фасоль'),
(4, 'Спаржа'),
(4, 'Капуста'),
(5, 'Кабачки'),
(5, 'Спаржа'),
(5, 'Фасоль'),
(5, 'Помидоры'),
(6, 'Помидоры'),
(6, 'Кукуруза'),
(7, 'Капуста'),
(7, 'Помидоры'),
(7, 'Перец'),
(8, 'Кабачки'),
(8, 'Спаржа'),
(8, 'Фасоль'),
(9, 'Фасоль'),
(9, 'Кукуруза'),
(10, 'Перец'),
(10, 'Капуста'),
(10, 'Фасоль'),
(10, 'Кабачки'),
(11, 'Спаржа'),
(11, 'Фасоль'),
(11, 'Кабачки'),
(12, 'Кабачки'),
(12, 'Кукуруза'),
(12, 'Спаржа'),
(12, 'Фасоль'),
(13, 'Кукуруза'),
(13, 'Перец'),
(13, 'Помидоры'),
(13, 'Фасоль'),
(13, 'Капуста')]

    user_ids, titles = zip(*data)
    user_ids = list(set(user_ids))
    titles = list(set(titles))

    df = pd.DataFrame(index=user_ids, columns=titles)
    for tr in data:
        uid, title = tr
        df[title][uid] = 1
    df = df.fillna(0)
    print(df)
    # Assosiation rule learning
    arl = MyARL()
    arl.apriori(df.values, min_support=0.15, min_confidence=0.6, labels=df.columns)

    antecedents, consequents, supports, confidences, lifts = zip(*arl.get_rules())
    itemsets, it_supports = zip(*arl.get_popular_itemsets())

    # Formatting and output
    rules_df = pd.DataFrame(data={
        "Antecedent": antecedents,
        "Consequent": consequents,
        "Support": supports,
        "Confidence": confidences,
        "Lift": lifts
    })
    rules_df.index += 1
    rules_df.to_csv('rules.csv')

    popular_itemsts_df = pd.DataFrame(data={
        'Itemset': itemsets,
        'Support': it_supports
    })
    popular_itemsts_df.index += 1
    popular_itemsts_df.to_csv('popular_itemsets.csv')
    


if __name__ == '__main__':
    main()