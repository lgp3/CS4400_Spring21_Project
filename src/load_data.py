import os

import editdistance
import pandas

# This file takes in the original CSVs and outputs the blocked pairs
# that can be used for feature engineering.

DATA_PATH = os.path.join(os.getcwd(), os.pardir, "data")

L_TABLE_PATH = os.path.join(DATA_PATH, "ltable.csv")
R_TABLE_PATH = os.path.join(DATA_PATH, "rtable.csv")
ALL_PAIRS_PATH = os.path.join(DATA_PATH, "all_pairs_bb.csv")
TRUTH_PATH = os.path.join(DATA_PATH, "train.csv")
TRAIN_PATH = os.path.join(DATA_PATH, "training_data_bb.csv")
TRAIN_PATH_COS_UPDATE = os.path.join(DATA_PATH, "training_data_cos.csv")
TRAIN_PARTIAL_PATH = os.path.join(DATA_PATH, "training_data_temp.csv")

NO_MATCH_CONSTANT = pandas.NA


def load_data_into_csv(ltable=L_TABLE_PATH,
                       rtable=R_TABLE_PATH,
                       train=TRUTH_PATH):
    ltable_df = pandas.read_csv(ltable)
    rtable_df = pandas.read_csv(rtable)
    train_df = pandas.read_csv(train)

    return ltable_df, rtable_df, train_df


def create_pairs(l_df, r_df, save=False):
    brand_column = 'brand'
    cat_column = 'category'

    l_df_brand = l_df.copy(deep=True)
    r_df_brand = r_df.copy(deep=True)

    l_df_brand = l_df_brand.set_index(brand_column)
    r_df_brand = r_df_brand.set_index(brand_column)
    print("joining")
    all_pairs_brand = l_df_brand.join(
        r_df_brand.drop([cat_column], axis=1).rename(columns={'category_duplicate': cat_column}),
        how='outer', on=brand_column,
        lsuffix='_l', rsuffix='_r')

    all_pairs_brand = all_pairs_brand[all_pairs_brand.index.notnull()]
    all_pairs_brand.rename(columns={'brand': 'brand_l', 'brand_duplicate': 'brand_r'}, inplace=True)
    all_pairs_brand = all_pairs_brand[
        (all_pairs_brand['brand_r'].notna()) & (all_pairs_brand['brand_l'].notna()) |
        (all_pairs_brand['brand_r'].isna()) & (all_pairs_brand['brand_l'].isna())]

    l_df_cat = l_df.copy(deep=True)
    r_df_cat = r_df.copy(deep=True)

    l_df_cat.set_index(cat_column, inplace=True)
    r_df_cat.set_index(cat_column, inplace=True)

    all_pairs_cat = l_df_cat.join(
        r_df_cat.drop([brand_column], axis=1).rename(columns={'brand_duplicate': brand_column}),
        how='outer', on=cat_column,
        lsuffix='_l', rsuffix='_r')
    all_pairs_cat = all_pairs_cat[all_pairs_cat.index.notnull()]

    all_pairs_cat.rename(columns={cat_column: 'category_l', 'category_duplicate': 'category_r'}, inplace=True)
    all_pairs_cat = all_pairs_cat[
        (all_pairs_cat['category_r'].notna()) & (all_pairs_cat['category_l'].notna()) |
        (all_pairs_cat['category_r'].isna()) & (all_pairs_cat['category_l'].isna())
    ]

    print("Size of cat pairs: {}".format(len(all_pairs_cat)))
    print("Size of brand pairs: {}".format(len(all_pairs_brand)))

    all_pairs = pandas.concat([all_pairs_brand, all_pairs_cat])
    all_pairs.drop_duplicates()
    print("Found {} pairs after blocking".format(len(all_pairs)))
    if save:
        all_pairs.to_csv(ALL_PAIRS_PATH)
    return all_pairs


def get_edit_percent(string_a, string_b):
    ed = editdistance.eval(string_a, string_b)
    max_len = max(len(string_a), len(string_b))
    return (max_len - ed) / max_len


def replace_categories(ltable, rtable):

    l_cats = ltable.category.unique()
    r_cats = rtable.category.unique()
    r_mapping = dict()
    for r_cat in r_cats:
        if r_cat is None:
            r_mapping.update({r_cat: NO_MATCH_CONSTANT})
            continue
        max_match_score = 0
        max_match_cat = l_cats[0]
        for l_cat in l_cats:
            score = 0 if l_cat is None else \
                get_edit_percent(str(r_cat), str(l_cat))
            if score > max_match_score:
                max_match_score = score
                max_match_cat = l_cat

        if max_match_score < .8:
            r_mapping.update({r_cat: NO_MATCH_CONSTANT})
        else:
            print(max_match_score)
            r_mapping.update({r_cat: str(max_match_cat)})

    rtable['category_duplicate'] = rtable['category']
    rtable['category'].replace(r_mapping, inplace=True)
    return rtable


def replace_brands(ltable, rtable):

    l_cats = ltable.brand.unique()
    r_cats = rtable.brand.unique()
    r_mapping = dict()
    for r_cat in r_cats:
        if r_cat is None:
            r_mapping.update({r_cat: NO_MATCH_CONSTANT})
            continue
        max_match_score = 0
        max_match_cat = l_cats[0]
        for l_cat in l_cats:
            score = 0 if l_cat is None else \
                get_edit_percent(str(r_cat), str(l_cat))
            if score > max_match_score:
                max_match_score = score
                max_match_cat = l_cat

        if max_match_score < 0.33:
            r_mapping.update({r_cat: NO_MATCH_CONSTANT})
        else:
            r_mapping.update({r_cat: str(max_match_cat)})

    rtable['brand_duplicate'] = rtable['brand']
    rtable['brand'].replace(r_mapping, inplace=True)
    return rtable


if __name__ == "__main__":

    ltable, rtable, train = load_data_into_csv()

    print_metadata = False
    if print_metadata:
        print(ltable.category.unique())
        print(rtable.category.unique())

        ltable['id'] = ltable['id'].astype(int)
        rtable['id'] = rtable['id'].astype(int)
        train['ltable_id'] = train['ltable_id'].astype(int)
        train['rtable_id'] = train['rtable_id'].astype(int)

        train_matches = train[train['label'] == 1]
        num_brand_matches = 0
        with_diff_brands = pandas.DataFrame()
        for idx, row in train_matches.iterrows():
            l_id = row['ltable_id']
            r_id = row['rtable_id']
            r_brand = rtable[rtable['id'] == r_id]['brand'].iloc[0]
            l_brand = ltable[ltable['id'] == l_id]['brand'].iloc[0]

            if r_brand == l_brand:
                num_brand_matches += 1
            else:
                with pandas.option_context('display.max_rows', None,
                                       'display.max_columns', None):
                    print(rtable[rtable['id'] == r_id]['brand'].iloc[0])
                    print(ltable[ltable['id'] == l_id]['brand'].iloc[0])
                print("\n")
        print('Num same brand: {}'.format(num_brand_matches))
    rtable = replace_brands(ltable, rtable)
    rtable = replace_categories(ltable, rtable)
    all_pairs = create_pairs(ltable, rtable, save=False)

    train_matches = train[train['label'] == 1]
    num_possible = 0
    for idx, row in train_matches.iterrows():
        l_id = row['ltable_id']
        r_id = row['rtable_id']

        if l_id in all_pairs.id_l.values and r_id in all_pairs.id_r.values:
            num_possible += 1
    print(num_possible)