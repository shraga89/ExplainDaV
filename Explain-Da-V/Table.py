import pandas as pd
import config

class Table:
    CATEGORICAL_UPPER_BOUND = config.CATEGORICAL_UPPER_BOUND

    def __init__(self, df):
        self.table = df
        self.headers = list(df.columns)
        self.A = list(range(0, len(df.columns)))
        self.A_vecs = list()  # temp
        self.set_A_vec_naive()
        self.A_types = self.set_attribute_types(False)
        self.fix_types()
        self.projected_table = self.table
        self.projected_A = self.A
        self.r = self.table.index.to_numpy()
        self.r_vecs = list()  # temp
        self.set_r_vec_naive()

    def set_A_vec_naive(self):
        self.A_vecs = [self.table[col].unique() for col in self.headers]

    def set_r_vec_naive(self):
        self.r_vecs = self.projected_table.to_numpy()

    def update_A_vec(self, vecs):
        self.A_vecs = vecs

    def update_projected_table(self, attributes):
        self.projected_table = self.table.iloc[:, attributes]
        self.projected_A = attributes

    def set_attribute_types(self, use_profiler=False):
        if use_profiler:
            profiler = ProfileReport(T_prime.table)
            var = prof.get_description()['variables']
            #             print({i: v for i, v in enumerate(self.A)})
            return {i: var[v]['type'] for i, v in enumerate(self.table.columns)}
        else:
            return dict(zip(self.A, self.table.dtypes))

    def fix_types(self):
        is_change = False
        for a in self.A_types:
            col = self.table.iloc[:, a]
            cardinality = len(col.unique())
            # if cardinality == 2:
            #     self.table.iloc[:, a] = self.table.iloc[:, a].astype('bool')
            #     is_change = True
            # elif cardinality <= self.CATEGORICAL_UPPER_BOUND:
            if cardinality <= self.CATEGORICAL_UPPER_BOUND:
                self.table.iloc[:, a] = self.table.iloc[:, a].astype('category')
                is_change = True
            elif self.A_types[a] == 'object' and pd.api.types.is_string_dtype(col):
                self.table.iloc[:, a] = self.table.iloc[:, a].astype(str)
                is_change = True
        if is_change:
            self.A_types = self.set_attribute_types(False)

    def get_attributes_types(self, attribute_set):
        types = []
        for a in attribute_set:
            types.append(self.A_types[a])
        return types

    def get_attributes_names(self, attribute_set):
        return [self.headers[i] for i in attribute_set]

    def get_feature_like_attributes(self):
        return [a for a in self.A_types if self.A_types[a] not in ['object']]

    def get_row_by_id(self, row_id):
        return self.table.iloc[row_id, :]

    def get_rows_by_ids(self, row_ids):
        return self.table.iloc[row_ids, :]
