import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
import glob
import json
import os
from Table import Table
from Table_Pair import Table_Pair
import ml_utils
from Foofah import foofah
from Foofah.foofah_libs.operators import *
from Foofah.foofah_libs import operators
from Foofah.foofah_libs import operators as Op
from config import *
from sklearn.model_selection import train_test_split
import inspect
from sklearn.preprocessing import OneHotEncoder
import time
import func_timeout


def find_exact_attribute_match(T_pair: Table_Pair):
    return T_pair.sigma_A
    # T_A_vecs = T_pair.T.A_vecs
    # T_prime_A_vecs = T_pair.T_prime.A_vecs
    # sigma_A = T_pair.sigma_A
    # sigma_A_exact = list()
    # for c in sigma_A:
    #     T_A_vec = T_A_vecs[c[0]]
    #     T_prime_A_vec = T_prime_A_vecs[c[1]]
    #     if np.array_equal(T_A_vec, T_prime_A_vec):
    #         sigma_A_exact.append(c)
    # return sigma_A_exact


def find_attribute_match(T_pair: Table_Pair, one_to_one=True):
    T = T_pair.T
    T_prime = T_pair.T_prime
    is_record_match = True
    # if 'orig_index' not in T_prime.table.columns:
    #     is_record_match = False
    ##############
    # Some magic #
    ##############
    #### Temp:
    # print(T.A)
    # print(T_prime.A)
    if one_to_one:
        sigma_A = list()
        # print(T.headers)
        # print(T_prime.headers)
        for i_left, h_left in enumerate(T.headers):
            for i_right, h_right in enumerate(T_prime.headers):
                if h_left == h_right:
                    # if set(T.table[h_left]) == set(T_prime.table[h_right]):
                    sigma_A += [(i_left, i_right), ]
        # sigma_A = list(zip(T.A, T_prime.A))  # temp
    else:
        sigma_A = list(zip(T.A, T_prime.A))  # temp
    if 'orig_index' not in T_prime.table.columns:
        is_record_match = False
        sigma_A = list()
        # print(T.headers)
        # print(T_prime.headers)
        for i_left, h_left in enumerate(T.headers):
            for i_right, h_right in enumerate(T_prime.headers):
                if h_left == h_right:
                    if set(T.table[h_left]) == set(T_prime.table[h_right]):
                        sigma_A += [(i_left, i_right), ]
    ###
    # print(sigma_A)
    T_pair.update_attribute_match(sigma_A)

    sigma_A_exact = find_exact_attribute_match(T_pair)
    T_pair.update_exact_attribute_match(sigma_A_exact)

    return is_record_match


def find_exact_record_match(T_pair: Table_Pair):
    return T_pair.sigma_r
    # T_r_vecs = T_pair.T.r_vecs
    # T_prime_r_vecs = T_pair.T_prime.r_vecs
    # sigma_r = T_pair.sigma_r
    # sigma_r_exact = list()
    # for c in sigma_r:
    #     T_r_vec = T_r_vecs[c[0]]
    #     T_prime_r_vec = T_prime_r_vecs[c[1]]
    #     if np.array_equal(T_r_vec, T_prime_r_vec):
    #         sigma_r_exact.append(c)
    # return sigma_r_exact


def find_record_match(T_pair: Table_Pair, one_to_one=True):
    T = T_pair.T
    T_prime = T_pair.T_prime
    # T_r = T.projected_table
    # T_prime_r = T_prime.projected_table
    # T_r_ids = T.r
    # T_prime_r_ids = T_prime.r
    ##############
    # Some magic #
    ##############
    #### Temp:
    # sigma_r = list(zip(T_r.dropna().index.to_numpy(), T_prime_r.index.to_numpy()))  # temp
    # print('!!!remember that the matching is only for removed rows!!!')
    if one_to_one:
        # sigma_r = list(zip(T_prime.table['orig_index'].to_numpy(), T_prime.table.index.to_numpy()))
        merged = T.table.merge(T_prime.table, left_index=True, right_on=['orig_index'], how='inner')
        sigma_r = list(zip(merged['orig_index'].to_numpy(), merged.index.to_numpy()))
    else:
        sigma_r = []
    ###
    T_pair.update_record_match(sigma_r)
    sigma_r_exact = find_exact_record_match(T_pair)
    T_pair.update_exact_record_match(sigma_r_exact)
    return True


def get_cardinality(df, attributes):
    cardinalities = {}
    # print(attributes)
    target = attributes[-1]
    for a_i in attributes:
        col_cardinality = len(df.iloc[:, a_i].drop_duplicates())
        cardinalities[a_i] = col_cardinality
    cardinalities_no_target = {a: cardinalities[a] for a in cardinalities if a != target}
    sorted_by_cardinalities = sorted(cardinalities_no_target, key=cardinalities_no_target.get)
    return sorted_by_cardinalities, cardinalities


def find_estimated_keys(df):
    full_size = len(df)
    keys = []
    print(full_size)
    for a_i, a in enumerate(df.columns):
        col_cardinality = len(df.iloc[:, a_i].drop_duplicates())
        print(a, col_cardinality)
        if col_cardinality == full_size:
            keys.append(a_i)
    return keys


def prioritize_attribute_sets(candidate_orig_attribute_sets, df):
    print('Remember to look at that')
    for a_s in candidate_orig_attribute_sets:
        print(a_s)
        print(len(df.iloc[:, a_s].drop_duplicates()))
    return candidate_orig_attribute_sets


def parse_fds(target_attribute):
    result_folder = glob.glob('./results/*')
    result_file = None
    if result_folder:
        result_file = max(result_folder, key=os.path.getctime)
    candidate_orig_attribute_sets = []
    if not result_file:
        print('could not find an FD for', target_attribute)
        return False
    with open(result_file, 'r') as file:
        for line in file:
            fdDict = json.loads(line)
            determinants = fdDict["determinant"]["columnIdentifiers"]
            dependant = fdDict["dependant"]
            #             print(determinants)
            #             print(dependant)
            #             print('--')
            tableName = dependant["tableIdentifier"]
            rhs = dependant["columnIdentifier"]
            # print([int(i["columnIdentifier"]) for i in determinants])
            # print(rhs)
            # print()
            if rhs == str(target_attribute):
                lhs = [int(i["columnIdentifier"]) for i in determinants]
                if len(lhs) <= max_size_lhd_fd:
                    candidate_orig_attribute_sets.append(lhs)
    remove_temp = Path(result_file)
    remove_temp.unlink()
    return candidate_orig_attribute_sets


def run_fdep(df):
    Path("./temp/").mkdir(parents=True, exist_ok=True)
    temp_name = 'temp/temp.csv'
    df.to_csv(temp_name, index=False)
    subprocess.call(['java',
                     '-cp',
                     'metanome.jar',
                     'de.metanome.cli.App',
                     '--algorithm', 'de.metanome.algorithms.fdep.FdepAlgorithmHashValues',
                     # '--algorithm', 'de.metanome.algorithms.tane.TaneAlgorithm',
                     # '--algorithm', 'de.metanome.algorithms.dfd.DFDMetanome',
                     '--files', '\"' + temp_name + '\"',
                     '--file-key', '\"Relational_Input\"'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    remove_temp = Path(temp_name)
    remove_temp.unlink()
    target_attribute = len(df.columns) - 1
    # print(df.columns[-1])
    #     print(target_attribute)
    candidate_orig_attribute_sets = parse_fds(target_attribute)
    return candidate_orig_attribute_sets


def find_fd(T_pair: Table_Pair, target_attribute, target_table='R', subset_of_origin=None):
    if target_table == 'R':
        avilable_df_attributes = T_pair.RHCA
        avilable_df_rows = T_pair.RHCr
        # avilable_df = T_pair.T_prime.table.iloc[avilable_df_rows, avilable_df_attributes]
        # target_attribute_values = T_pair.T_prime.table.iloc[:, target_attribute]
        combined_df = T_pair.T_prime.table.iloc[avilable_df_rows, avilable_df_attributes + [target_attribute, ]]
    else:
        if subset_of_origin:
            avilable_df_attributes = subset_of_origin
        else:
            avilable_df_attributes = T_pair.LHCA
        avilable_df_rows = T_pair.LHCr
        # estimated_keys = find_estimated_keys(T_pair.T.table)
        # print(estimated_keys)
        # avilable_df = T_pair.T.table.iloc[avilable_df_rows, avilable_df_attributes]
        # target_attribute_values = T_pair.T.table.iloc[:, target_attribute]
        combined_df = T_pair.T.table.iloc[avilable_df_rows, avilable_df_attributes + [target_attribute, ]]
        # print(combined_df)
        # combined_df = T_pair.T.table.iloc[avilable_df_rows, [0, 1, 8]]
        # print(combined_df)
    #     print(avilable_df_attributes)
    #     print(len(combined_df['country'].unique()))
    #     print(len(combined_df['country_abbv'].unique()))
    candidate_orig_attribute_sets = run_fdep(combined_df)
    if candidate_orig_attribute_sets:
        return candidate_orig_attribute_sets
    elif subset_of_origin:
        return []
    else:
        # return [avilable_df_attributes]
        return []


def get_target_type(_type):
    target_task = None
    if pd.api.types.is_bool_dtype(_type):
        target_task = 'BINARY_CLASSIFICATION'
    elif pd.api.types.is_categorical_dtype(_type):
        target_task = 'MULTICLASS_CLASSIFICATION'
    elif pd.api.types.is_numeric_dtype(_type):
        target_task = 'REGRESSION'
    else:
        target_task = 'TEXTUAL'
    return target_task


def get_task_type_and_solve(T_pair_validation,
                            T_pair_generalization,
                            attribute_set,
                            attribute_2_b_explained,
                            LHS_types,
                            target_task,
                            run_only_reg=False):
    try:
        X_train = T_pair_validation.T_prime.projected_table.iloc[:, attribute_set]
        X_test = T_pair_generalization.T_prime.projected_table.iloc[:, attribute_set]
    except:
        X_train = T_pair_validation.T_prime.table.iloc[:, attribute_set]
        X_test = T_pair_generalization.T_prime.table.iloc[:, attribute_set]
    y_train = T_pair_validation.T_prime.table.iloc[:, attribute_2_b_explained]
    y_test = T_pair_generalization.T_prime.table.iloc[:, attribute_2_b_explained]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_for_cv, random_state=42)
    solutions = {}
    if any([pd.api.types.is_string_dtype(_type) for _type in LHS_types]):
        features_task = 'TEXTUAL'
    else:
        features_task = 'NUMERIC'
    print('Features:', features_task, '| Target:', target_task)
    # print(run_only_reg)
    if run_only_reg and features_task != 'NUMERIC':
        print('skipping (only regression)')
        return solutions
    if (target_task == 'TEXTUAL') and (features_task == 'TEXTUAL'):
        sols = data_transformation_detection(X_train,
                                             y_train,
                                             X_test,
                                             y_test)
        if sols:
            solutions.update(sols)
    elif (target_task != 'TEXTUAL') and (features_task == 'TEXTUAL'):
        if target_task == 'BINARY_CLASSIFICATION':
            y_train = ml_utils.fix_labels(y_train)
            y_test = ml_utils.fix_labels(y_test)
        sols, new_feature_names = learn_data_transformation_textual_feature(X_train,
                                                                            y_train,
                                                                            X_test,
                                                                            y_test,
                                                                            target_task,
                                                                            grouping=False)
        if len(sols) != 0:
            # print(sols)
            for i, a_i in reversed(list(enumerate(new_feature_names))):
                sols = {sol.replace('x' + str(i), a_i): sols[sol] for sol in sols}
            for i, a_i in enumerate(T_pair_validation.T_prime.get_attributes_names(attribute_set)):
                sols = {sol.replace('x' + str(i), a_i): sols[sol] for sol in sols}
            perfect_solution = list(sols.values())[0][0] == list(sols.values())[0][0] == 1.0
            solutions.update(sols)
        else:
            perfect_solution = False
        sols, _ = learn_data_transformation_textual_feature(X_train,
                                                            y_train,
                                                            X_test,
                                                            y_test,
                                                            target_task,
                                                            grouping=True)
        if len(sols) != 0:
            perfect_solution = list(sols.values())[0][0] == list(sols.values())[0][0] == 1.0
            solutions.update(sols)
        else:
            perfect_solution = False
        if not perfect_solution:
            new_sols = data_transformation_detection(X_train,
                                                     y_train,
                                                     X_test,
                                                     y_test,
                                                     target_type='NUMERIC')
            if new_sols:
                solutions.update(new_sols)
    elif (target_task == 'TEXTUAL') and (features_task != 'TEXTUAL'):
        sols = data_transformation_detection(X_train,
                                             y_train,
                                             X_test,
                                             y_test)
        if sols:
            solutions.update(sols)
    else:
        try:
            X_train = ml_utils.fix_domains(X_train)
            X_test = ml_utils.fix_domains(X_test)
        except:
            print('did not fix domain')
        try:
            if target_task != 'REGRESSION':
                y_train = ml_utils.fix_labels(y_train)
                y_test = ml_utils.fix_labels(y_test)
        except:
            print('did not fix labels')
        y_train = np.nan_to_num(y_train)
        y_test = np.nan_to_num(y_test)
        try:
            sols = learn_data_transformation(X_train, y_train, X_test, y_test, target_task)
        except:
            X_train = X_train.fillna(0.0)
            X_train = X_train.mask(X_train > new_inf, new_inf)
            X_train = X_train.mask(X_train < -new_inf, -new_inf)
            X_test = X_test.fillna(0.0)
            X_test = X_test.mask(X_test > new_inf, new_inf)
            X_test = X_test.mask(X_test < -new_inf, -new_inf)
            sols = learn_data_transformation(X_train, y_train, X_test, y_test, target_task)
        for i, a_i in enumerate(T_pair_validation.T_prime.get_attributes_names(attribute_set)):
            sols = {sol.replace('x' + str(i), a_i): sols[sol] for sol in sols}
        if sols:
            solutions.update(sols)
        sols = learn_data_transformation(X_train, y_train, X_test, y_test, target_task, True)
        for i, a_i in enumerate(T_pair_validation.T_prime.get_attributes_names(attribute_set)):
            sols = {sol.replace('x' + str(i), a_i): sols[sol] for sol in sols}
        if sols:
            solutions.update(sols)
    return solutions


def get_task_type_and_solve_baseline(T_pair_validation,
                                     T_pair_generalization,
                                     attribute_set,
                                     attribute_2_b_explained,
                                     LHS_types,
                                     target_task,
                                     extend_for_auto_pipeline=False,
                                     extend_for_plus=False):
    try:
        X_train = T_pair_validation.T_prime.projected_table.iloc[:, attribute_set]
        X_test = T_pair_generalization.T_prime.projected_table.iloc[:, attribute_set]
    except:
        X_train = T_pair_validation.T_prime.table.iloc[:, attribute_set]
        X_test = T_pair_generalization.T_prime.table.iloc[:, attribute_set]
    y_train = T_pair_validation.T_prime.table.iloc[:, attribute_2_b_explained]
    y_test = T_pair_generalization.T_prime.table.iloc[:, attribute_2_b_explained]
    solutions = {}
    if extend_for_auto_pipeline:
        sols = None
        if target_task not in ['REGRESSION', 'BINARY_CLASSIFICATION']:
            sols = data_transformation_detection(X_train,
                                                 y_train,
                                                 X_test,
                                                 y_test,
                                                 what_to_explain='auto-pipeline')
        else:
            print(target_task)
            print('skipping')
        if sols:
            solutions.update(sols)
    elif extend_for_plus:
        sols = data_transformation_detection(X_train,
                                             y_train,
                                             X_test,
                                             y_test,
                                             what_to_explain='foofah_plus')
        if sols:
            solutions.update(sols)
    else:
        sols = None
        if target_task not in ['REGRESSION', 'BINARY_CLASSIFICATION']:
            sols = data_transformation_detection(X_train,
                                                 y_train,
                                                 X_test,
                                                 y_test,
                                                 what_to_explain='foofah')
        else:
            print(target_task)
            print('skipping')
        if sols:
            solutions.update(sols)

    return solutions


def resolve_for_attribute_set(T_pair_validation,
                              T_pair_generalization,
                              attribute_set,
                              attribute_2_b_explained,
                              target_task,
                              run_only_reg=False):
    # print('using', attribute_set, '(', T_pair.T_prime.get_attributes_names(attribute_set), ')')
    LHS_types = T_pair_validation.T_prime.get_attributes_types(attribute_set)
    solutions = get_task_type_and_solve(T_pair_validation,
                                        T_pair_generalization,
                                        attribute_set,
                                        attribute_2_b_explained,
                                        LHS_types,
                                        target_task,
                                        run_only_reg=run_only_reg)
    return solutions


def resolve_for_attribute_set_baseline(T_pair_validation,
                                       T_pair_generalization,
                                       attribute_set,
                                       attribute_2_b_explained,
                                       target_task,
                                       extend_for_auto_pipeline=False,
                                       extend_for_plus=False):
    # print('using', attribute_set, '(', T_pair.T_prime.get_attributes_names(attribute_set), ')')
    LHS_types = T_pair_validation.T_prime.get_attributes_types(attribute_set)
    solutions = get_task_type_and_solve_baseline(T_pair_validation,
                                                 T_pair_generalization,
                                                 attribute_set,
                                                 attribute_2_b_explained,
                                                 LHS_types,
                                                 target_task,
                                                 extend_for_auto_pipeline=extend_for_auto_pipeline,
                                                 extend_for_plus=extend_for_plus)
    return solutions


def data_transformation_detection(X_train,
                                  y_train,
                                  X_test,
                                  y_test,
                                  what_to_explain='columns',
                                  target_type='TEXTUAL'):
    start = time.time()
    # print('Using Foofah')
    # if 'variety' not in str(pd.DataFrame(X_train).columns):
    #     return None, None, None, None, None, None
    foofah_input = '{"InputTable": '
    foofah_input += str(X_train.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "OutputTable": '
    # ROEE: Only required column:
    # foofah_input += str(pd.DataFrame(y.head(number_of_rows_for_foofah)).to_json(orient="values"))
    # ROEE: Required column + Context:
    # print(X_train)
    # print(y_train)
    if what_to_explain == 'full':
        foofah_input += str(y_train.head(number_of_rows_for_foofah).to_json(orient="values"))
    else:
        try:
            foofah_input += str(
                pd.DataFrame(X_train.join(y_train).head(number_of_rows_for_foofah)).to_json(orient="values"))
        except:
            y_train = y_train.astype(str)
            foofah_input += str(y_train.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "NumSamples": 1'
    # foofah_input += ', "TestName": "orig_{}_target_{}"'.format(str(pd.DataFrame(X_train).columns.tolist()),
    #                                                            str(y_train.name))
    foofah_input += ', "TestName": "1"'
    foofah_input += ', "TestingTable": '
    foofah_input += str(X_test.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "TestAnswer": '
    if what_to_explain == 'full':
        foofah_input += str(y_test.head(number_of_rows_for_foofah).to_json(orient="values"))
        foofah_input += '}'
    else:
        try:
            foofah_input += str(
                pd.DataFrame(X_test.join(y_test).head(number_of_rows_for_foofah)).to_json(orient="values"))
            foofah_input += '}'
        except:
            y_test = y_test.astype(str)
            foofah_input += str(y_test.head(number_of_rows_for_foofah).to_json(orient="values"))
            foofah_input += '}'
    # print(foofah_input)
    with open('task_for_foofah.txt', 'w') as f:
        f.write(foofah_input)
    # print(str(X.head(number_of_rows_for_foofah).to_json(orient="values")))
    # print(str(y.head(number_of_rows_for_foofah).to_json(orient="values")))
    # print(str(pd.DataFrame(X_train.join(y_train).head(number_of_rows_for_foofah)).to_json(orient="values")))
    is_solved = foofah.main('task_for_foofah.txt', foofah_time_limit, what_to_explain, target_type, True)
    # with open('task_for_foofah.txt', 'rb') as f:
    #     test_data = json.load(f)
    # raw_data = [list(map(str, x)) for x in test_data['InputTable']]
    # print(raw_data)
    # print(type(X))
    # print(X.columns)
    eval_explainabilty_size = 0
    eval_explainabilty_cognitive_chunks = 1
    seen_ops = []
    if is_solved:
        transformation = 't = ' + ','.join(list(X_train.columns)) + '\n'
        with open('foo.txt', 'r') as f:
            for op in f.readlines():
                transformation += op
                eval_explainabilty_cognitive_chunks += 1
                op_striped = op.split('(')[0].split(' = ')[1]
                # if op_striped not in seen_ops:
                #     seen_ops.append(op_striped)
                seen_ops.append(op_striped)
                op_lines = inspect.getsource(getattr(operators, op_striped)).splitlines()
                eval_explainabilty_size += len([code_line for code_line in op_lines
                                                if len(code_line.replace(' ', '')) and
                                                code_line.replace(' ', '')[0] != '#'])

                # f, param = op.split(' ', 1)
                # param = [raw_data, ] + [int(p) if p.isnumeric() else p
                #                         for p in param.replace('\n', '').split(',')]
                # print(f)
                # print(param)
                # Op.PRUNE_1 = False
                # print(f_split_first(raw_data, 0, ' '))
                # print(globals()[f](*param))
        # print(transformation)
        eval_explainabilty_repeated_terms = len(seen_ops)
        eval_explainabilty_cognitive_chunks = len(X_train.columns)
        eval_validation = 1.0
        with open('test_results/validate/exp0_results_1_1.txt', 'r') as f:
            eval_generalization = float(bool(dict(json.load(f))['Success']))
        # eval_generalization = 1.0
        # eval_simplicity = 0.9
        eval_explainabilty = [eval_explainabilty_size,
                              eval_explainabilty_repeated_terms,
                              eval_explainabilty_cognitive_chunks]
        run_time = time.time() - start
        return {transformation: [eval_validation, eval_generalization, *eval_explainabilty, run_time]}
    else:
        return None
    # foofah.main('examples/exp0_agriculture_5.txt')
    # print('To-Be-Implemented')
    # print('-- Since we have textual values in both the target'
    #       ' and features, perform \"standard\" data transformation detection')


def data_transformation_detection_column_removal(X_train,
                                                 y_train,
                                                 X_test,
                                                 y_test,
                                                 what_to_explain='columns',
                                                 target_type='TEXTUAL'):
    start = time.time()
    foofah_input = '{"InputTable": '
    foofah_input += str(X_train.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "OutputTable": '
    foofah_input += str(y_train.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "NumSamples": 1'
    foofah_input += ', "TestName": "1"'
    foofah_input += ', "TestingTable": '
    foofah_input += str(X_test.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "TestAnswer": '
    foofah_input += str(y_test.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += '}'
    with open('task_for_foofah.txt', 'w') as f:
        f.write(foofah_input)
    is_solved = foofah.main('task_for_foofah.txt', foofah_time_limit, what_to_explain, target_type, True)
    eval_explainabilty_size = 0
    eval_explainabilty_cognitive_chunks = 1
    seen_ops = []
    if is_solved:
        transformation = 't = ' + ','.join(list(X_train.columns)) + '\n'
        with open('foo.txt', 'r') as f:
            for op in f.readlines():
                transformation += op
                eval_explainabilty_cognitive_chunks += 1
                op_striped = op.split('(')[0].split(' = ')[1]
                seen_ops.append(op_striped)
                op_lines = inspect.getsource(getattr(operators, op_striped)).splitlines()
                eval_explainabilty_size += len([code_line for code_line in op_lines
                                                if len(code_line.replace(' ', '')) and
                                                code_line.replace(' ', '')[0] != '#'])
        eval_explainabilty_repeated_terms = len(seen_ops)
        eval_explainabilty_cognitive_chunks = len(X_train.columns)
        eval_validation = 1.0
        with open('test_results/validate/exp0_results_1_1.txt', 'r') as f:
            eval_generalization = float(bool(dict(json.load(f))['Success']))
        eval_explainabilty = [eval_explainabilty_size,
                              eval_explainabilty_repeated_terms,
                              eval_explainabilty_cognitive_chunks]
        run_time = time.time() - start
        return {transformation: [eval_validation, eval_generalization, *eval_explainabilty, run_time]}
    else:
        return None


def data_transformation_detection_rows_removal(X_train,
                                               y_train,
                                               X_test,
                                               y_test,
                                               what_to_explain='columns',
                                               target_type='TEXTUAL'):
    start = time.time()
    foofah_input = '{"InputTable": '
    foofah_input += str(X_train.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "OutputTable": '
    foofah_input += str(y_train.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "NumSamples": 1'
    foofah_input += ', "TestName": "1"'
    foofah_input += ', "TestingTable": '
    foofah_input += str(X_test.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "TestAnswer": '
    foofah_input += str(y_test.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += '}'
    with open('task_for_foofah.txt', 'w') as f:
        f.write(foofah_input)
    is_solved = foofah.main('task_for_foofah.txt', foofah_time_limit, what_to_explain, target_type, True)
    eval_explainabilty_size = 0
    eval_explainabilty_cognitive_chunks = 1
    seen_ops = []
    if is_solved:
        transformation = 't = ' + ','.join(list(X_train.columns)) + '\n'
        with open('foo.txt', 'r') as f:
            for op in f.readlines():
                transformation += op
                eval_explainabilty_cognitive_chunks += 1
                op_striped = op.split('(')[0].split(' = ')[1]
                seen_ops.append(op_striped)
                op_lines = inspect.getsource(getattr(operators, op_striped)).splitlines()
                eval_explainabilty_size += len([code_line for code_line in op_lines
                                                if len(code_line.replace(' ', '')) and
                                                code_line.replace(' ', '')[0] != '#'])
        eval_explainabilty_repeated_terms = len(seen_ops)
        eval_explainabilty_cognitive_chunks = len(X_train.columns)
        eval_validation = 1.0
        with open('test_results/validate/exp0_results_1_1.txt', 'r') as f:
            eval_generalization = float(bool(dict(json.load(f))['Success']))
        eval_explainabilty = [eval_explainabilty_size,
                              eval_explainabilty_repeated_terms,
                              eval_explainabilty_cognitive_chunks]
        run_time = time.time() - start
        return {transformation: [eval_validation, eval_generalization, *eval_explainabilty, run_time]}
    else:
        return None


def data_transformation_detection_rows_addition(X_train,
                                                y_train,
                                                X_test,
                                                y_test,
                                                what_to_explain='columns',
                                                target_type='TEXTUAL'):
    start = time.time()
    foofah_input = '{"InputTable": '
    foofah_input += str(X_train.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "OutputTable": '
    foofah_input += str(y_train.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "NumSamples": 1'
    foofah_input += ', "TestName": "1"'
    foofah_input += ', "TestingTable": '
    foofah_input += str(X_test.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += ', "TestAnswer": '
    foofah_input += str(y_test.head(number_of_rows_for_foofah).to_json(orient="values"))
    foofah_input += '}'
    with open('task_for_foofah.txt', 'w') as f:
        f.write(foofah_input)
    is_solved = foofah.main('task_for_foofah.txt', foofah_time_limit, what_to_explain, target_type, True)
    eval_explainabilty_size = 0
    eval_explainabilty_cognitive_chunks = 1
    seen_ops = []
    if is_solved:
        transformation = 't = ' + ','.join(list(X_train.columns)) + '\n'
        with open('foo.txt', 'r') as f:
            for op in f.readlines():
                transformation += op
                eval_explainabilty_cognitive_chunks += 1
                op_striped = op.split('(')[0].split(' = ')[1]
                seen_ops.append(op_striped)
                op_lines = inspect.getsource(getattr(operators, op_striped)).splitlines()
                eval_explainabilty_size += len([code_line for code_line in op_lines
                                                if len(code_line.replace(' ', '')) and
                                                code_line.replace(' ', '')[0] != '#'])
        eval_explainabilty_repeated_terms = len(seen_ops)
        eval_explainabilty_cognitive_chunks = len(X_train.columns)
        eval_validation = 1.0
        with open('test_results/validate/exp0_results_1_1.txt', 'r') as f:
            eval_generalization = float(bool(dict(json.load(f))['Success']))
        eval_explainabilty = [eval_explainabilty_size,
                              eval_explainabilty_repeated_terms,
                              eval_explainabilty_cognitive_chunks]
        run_time = time.time() - start
        return {transformation: [eval_validation, eval_generalization, *eval_explainabilty, run_time]}
    else:
        return None


def feature_generation_and_then_transformation_detection():
    print('To-Be-Implemented')
    # print('-- Since we have some textual features are numeric'
    #       ' and/or categorial target, perform feature generation'
    #       ' to detect origin and then \"standard\" data transformation detection')


def learn_data_transformation(X_train,
                              y_train,
                              X_test,
                              y_test,
                              y_type,
                              only_regression=False,
                              binary_features=None,
                              include_aggregated=False,
                              non_numeric=None):
    # print(y_train)
    if only_regression:
        # print('-- Solving (all) as REGRESSION')
        return ml_utils.learn_data_transformation_REGRESSION(X_train,
                                                             y_train,
                                                             X_test,
                                                             y_test,
                                                             binary_features,
                                                             include_aggregated,
                                                             non_numeric)
    if y_type == 'BINARY_CLASSIFICATION':
        # print('-- Solving as BINARY_CLASSIFICATION')
        return ml_utils.learn_data_transformation_BINARY_CLASSIFICATION(X_train, y_train, X_test, y_test)
    elif y_type == 'MULTICLASS_CLASSIFICATION':
        # print('-- Solving as MULTICLASS_CLASSIFICATION')
        return ml_utils.learn_data_transformation_MULTICLASS_CLASSIFICATION(X_train, y_train, X_test, y_test,
                                                                            include_aggregated,
                                                                            non_numeric)
    elif y_type == 'REGRESSION':
        # print('-- Solving as REGRESSION')
        return ml_utils.learn_data_transformation_REGRESSION(X_train,
                                                             y_train,
                                                             X_test,
                                                             y_test,
                                                             binary_features,
                                                             include_aggregated,
                                                             non_numeric)
    else:
        print('Unknown Task Type')


def learn_data_transformation_textual_feature(X_train, y_train, X_test, y_test, y_type, grouping=False):
    categorical_attributes = [a for a in X_train.columns if pd.api.types.is_categorical_dtype(X_train[a])]
    numeric_attributes = [a for a in X_train.columns if not pd.api.types.is_string_dtype(X_train[a]) and
                          a not in categorical_attributes]
    textual_attributes_with_small_domain = [a for a in X_train.columns if pd.api.types.is_string_dtype(X_train[a]) and
                                            len(X_train[a].unique()) <
                                            textual_attributes_with_small_domain_threshold]
    textual_attributes_with_small_domain += categorical_attributes
    if len(numeric_attributes) + len(textual_attributes_with_small_domain) == 0:
        return {}, []
    X_train_categorial = X_train[textual_attributes_with_small_domain]
    X_train_new = X_train[numeric_attributes]
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train_categorial)
    one_hot_rep = pd.DataFrame(ohe.transform(X_train_categorial).toarray(), columns=ohe.get_feature_names())
    X_train_new[one_hot_rep.columns.tolist()] = one_hot_rep
    try:
        X_train_new = X_train_new.fillna(0)
    except:
        X_train_new = X_train_new

    X_test_categorial = X_test[textual_attributes_with_small_domain]
    X_test_new = X_test[numeric_attributes]
    try:
        one_hot_rep = pd.DataFrame(ohe.transform(X_test_categorial).toarray(), columns=ohe.get_feature_names())
    except:
        print('cant one hot')
        return {}, []

    X_test_new[one_hot_rep.columns.tolist()] = one_hot_rep
    try:
        X_test_new = X_test_new.fillna(0)
    except:
        X_test_new = X_test_new

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # print(numeric_attributes)
    # print(X_train_new.head().to_string())
    if not grouping:
        try:
            learn_data_transformation_lambda = lambda: learn_data_transformation(X_train_new, y_train, X_test_new,
                                                                                 y_test, y_type,
                                                                                 binary_features=one_hot_rep.columns.tolist())
            new_solutions = run_function(learn_data_transformation_lambda, foofah_time_limit)
            # print(new_solutions)
            return new_solutions, X_train.columns.tolist()
        except:
            return {}, []
    elif numeric_attributes:
        try:
            learn_data_transformation_lambda = lambda:  learn_data_transformation(X_train, y_train, X_test, y_test, y_type,
                                               include_aggregated=True,
                                               non_numeric=textual_attributes_with_small_domain,
                                               only_regression=True)
            result = run_function(learn_data_transformation_lambda, foofah_time_limit)
        except:
            return {}, None
        return result, None
    else:
        return {}, None


def run_function(f, max_wait):
    try:
        return func_timeout.func_timeout(max_wait, f)
    except func_timeout.FunctionTimedOut:
        print('timeout')
        return {}
    except:
        print('other error')
        return {}


def inspect_row(row_id, T_pair_validation):
    row_to_inspect_validation = T_pair_validation.T.get_row_by_id(row_id)
    row_solution = table_independent_row_inspection(row_to_inspect_validation)
    if not row_solution:
        row_solution = table_dependent_row_inspection(row_to_inspect_validation,
                                                      T_pair_validation.T_prime.projected_table)
    # Conflict with predicate resolution:
    # if not row_solution:
    #     row_solution = table_dependent_row_inspection_by_column(row_to_inspect, T_pair.T_prime.projected_table)
    return row_solution


def inspect_multiple_rows(unsolved_rows_validation,
                          T_pair_validation,
                          unsolved_rows_generalization,
                          T_pair_generalization):
    full_rows_validation = T_pair_validation.T.get_rows_by_ids(unsolved_rows_validation)
    full_table_validation = T_pair_validation.T_prime.projected_table
    full_rows_generalization = T_pair_generalization.T.get_rows_by_ids(unsolved_rows_generalization)
    full_table_generalization = T_pair_generalization.T_prime.projected_table

    categorical_attributes = [a for a, a_type in
                              zip(T_pair_validation.T.projected_A,
                                  T_pair_validation.T.get_attributes_types(T_pair_validation.T.projected_A)) if
                              pd.api.types.is_categorical_dtype(a_type)]

    numeric_attributes = [a for a, a_type in
                          zip(T_pair_validation.T.projected_A,
                              T_pair_validation.T.get_attributes_types(T_pair_validation.T.projected_A)) if
                          not pd.api.types.is_string_dtype(a_type) and a not in categorical_attributes]

    textual_attributes_with_small_domain = [a for a, a_type in
                                            zip(T_pair_validation.T.projected_A,
                                                T_pair_validation.T.get_attributes_types(
                                                    T_pair_validation.T.projected_A)) if
                                            pd.api.types.is_string_dtype(a_type) and
                                            len(T_pair_validation.T.table.iloc[:, a].unique()) <
                                            textual_attributes_with_small_domain_threshold] + categorical_attributes
    # rows_to_inspect = full_rows.iloc[:, numeric_attributes]
    # revised_table = full_table.iloc[:, numeric_attributes]
    # for a_small_domain in textual_attributes_with_small_domain:
    #     one_hot_rep_rows = pd.get_dummies(full_rows.iloc[:, a_small_domain],
    #                                      prefix=str(a_small_domain)).astype(int)
    #     rows_to_inspect = rows_to_inspect.join(one_hot_rep_rows)
    #     print(rows_to_inspect.columns)
    #     one_hot_rep_table = pd.get_dummies(full_table.iloc[:, a_small_domain],
    #                                      prefix=str(a_small_domain)).astype(int)
    #     revised_table = full_table.join(one_hot_rep_table)
    #     print(revised_table.columns)
    # print(T_pair_validation.T_prime.projected_table)

    return predicate_resolution(full_rows_validation, full_table_validation,
                                full_rows_generalization, full_table_generalization,
                                numeric_attributes, textual_attributes_with_small_domain)
    # TODO: NON numeric types predicate resolution


def table_independent_row_inspection(row):
    flag = False
    # print(' '.join(str(v) for v in row.tolist()))
    if row.isna().any() or 'nan' in ' '.join(str(v) for v in row.tolist()):
        flag = 'Columns_*{}*_CONTAIN/S_NAN'.format(','.join(pd.DataFrame(row[row.isna()]).index.tolist()))
        # print('?Do we also want to provide the column that had the null value?')
        # print('The row was removed because it contains a NaN value')
    return flag


def table_dependent_row_inspection(row, table):
    # print('--- TBI ---')
    flag = False
    expanded_table = table.append(row)
    if expanded_table.duplicated().tolist()[-1]:
        flag = 'DUPLICATED'
        # print('The row was removed as a result of deduplication')
    else:
        flag = 'UNKNOWN'
        # print('!!!apply predicate resolution!!!')
        # print('FIND MOST SIMILAR ROWS and then do that:')
        # data_transformation_detection(table, expanded_table, len(expanded_table), False)
    return flag


def get_recreation_operations(sols):
    op_set = set()
    for sol in sols:
        if 'DUPLICATED' in sols[sol]:
            op_set.add('REMOVE_DUPLICATED')
        if 'CONTAIN/S_NAN' in sols[sol]:
            op_set.add('DROP_NA')
        if len(op_set) == 2:
            return op_set
    return op_set


def evaluate_row_removal(T_pair_validation,
                         T_pair_generalization,
                         op_set,
                         validation_labels,
                         generalization_labels):
    start = time.time()
    recreated_T_prime_validation = T_pair_validation.T.table
    recreated_T_prime_generalization = T_pair_generalization.T.table
    # Basic Operations
    if 'REMOVE_DUPLICATED' in op_set:
        recreated_T_prime_validation = recreated_T_prime_validation.drop_duplicates()
        recreated_T_prime_generalization = recreated_T_prime_generalization.drop_duplicates()
    if 'REMOVE_DUPLICATED' in op_set:
        recreated_T_prime_validation = recreated_T_prime_validation.dropna()
        print('also remove those that contain nan')
        recreated_T_prime_generalization = recreated_T_prime_generalization.dropna()
    # Apply resolved predicate
    recreated_T_prime_validation['predicate_label'] = recreated_T_prime_validation.index
    recreated_T_prime_validation['predicate_label'] = recreated_T_prime_validation['predicate_label'].apply(lambda x:
                                                                                                            validation_labels.get(
                                                                                                                x))
    recreated_T_prime_validation = recreated_T_prime_validation[recreated_T_prime_validation['predicate_label'] == 0]

    recreated_T_prime_generalization['predicate_label'] = recreated_T_prime_generalization.index
    recreated_T_prime_generalization['predicate_label'] = recreated_T_prime_generalization['predicate_label'].apply(
        lambda x:
        generalization_labels.get(x))
    recreated_T_prime_generalization = recreated_T_prime_generalization[
        recreated_T_prime_generalization['predicate_label'] == 0]

    valid_ix = recreated_T_prime_validation.index.tolist()
    T_prime_valid_ix = T_pair_validation.T_prime.table.index.tolist()
    if len(valid_ix):
        overlap_valid = len(set(valid_ix).intersection(T_prime_valid_ix)) / float(len(valid_ix))
    else:
        overlap_valid = 0.0

    generalization_ix = recreated_T_prime_generalization.index.tolist()
    T_prime_generalization_ix = T_pair_generalization.T_prime.table.index.tolist()
    if len(generalization_ix):
        overlap_generalization = len(set(generalization_ix).intersection(T_prime_generalization_ix)) / float(
            len(generalization_ix))
    else:
        overlap_generalization = 0.0
    run_time = time.time() - start
    return [overlap_valid, overlap_generalization, 1.0, 1.0, 1.0, run_time]


def table_dependent_row_inspection_by_column(row, table):
    # print('--- TBI ---')
    flag = False
    # expanded_table = table.append(row)
    numeric_attributes = [a for a in table.columns if not pd.api.types.is_string_dtype(table[a])]
    for col in numeric_attributes:
        mean = np.mean(table[col])
        std = np.std(table[col])
        row_in_col = row[col].mean()
        z = (row_in_col - mean) / std
        # if z > 10:
        #     print(row_in_col)
        #     print(mean)
        #     flag = 'OUTLIER (Z-method)_(' + col + ')'
        #     break
        Q1 = np.percentile(table[col], 25, interpolation='midpoint')
        Q2 = np.percentile(table[col], 50, interpolation='midpoint')
        Q3 = np.percentile(table[col], 75, interpolation='midpoint')
        IQR = Q3 - Q1
        low_lim = Q1 - 1.5 * IQR
        up_lim = Q3 + 1.5 * IQR
        if row_in_col > up_lim or row_in_col < low_lim:
            flag = 'OUTLIER (IQR-method)_(' + col + ')'
            break
    return flag


def predicate_resolution(full_rows_validation, full_table_validation,
                         full_rows_generalization, full_table_generalization,
                         numeric_attributes, textual_attributes_with_small_domain):
    X_0 = full_table_validation
    y_0 = [0] * len(full_table_validation)
    X_1 = full_rows_validation
    y_1 = [1] * len(full_rows_validation)
    X_train_before_projection = pd.concat([X_0, X_1])
    X_train = X_train_before_projection.iloc[:, numeric_attributes]
    X_train_before_projection_categorial = X_train_before_projection.iloc[:,
                                           textual_attributes_with_small_domain].astype(str)
    ohe = OneHotEncoder(handle_unknown='ignore')
    # print(textual_attributes_with_small_domain)
    # print(full_table_validation)
    ohe.fit(X_train_before_projection_categorial)
    one_hot_rep = pd.DataFrame(ohe.transform(X_train_before_projection_categorial).toarray(),
                               columns=ohe.get_feature_names())
    X_train[one_hot_rep.columns.tolist()] = one_hot_rep
    # print(one_hot_rep)
    # print(X_train)
    X_train = X_train.fillna(0)
    # if len(textual_attributes_with_small_domain):
    #     for a_small_domain in textual_attributes_with_small_domain:
    #         X_for_a = X_train_before_projection.iloc[:, a_small_domain]
    #         one_hot_rep = pd.get_dummies(X_for_a, prefix=str(a_small_domain)).astype(int)
    #         X_train[one_hot_rep.columns.tolist()] = one_hot_rep

    y_train = y_0 + y_1
    X_0 = full_table_generalization
    y_0 = [0] * len(full_table_generalization)
    X_1 = full_rows_generalization
    y_1 = [1] * len(full_rows_generalization)
    X_test_before_projection = pd.concat([X_0, X_1])
    X_test = X_test_before_projection.iloc[:, numeric_attributes]
    X_test_before_projection_categorial = X_test_before_projection.iloc[:,
                                          textual_attributes_with_small_domain].astype(str)
    one_hot_rep = pd.DataFrame(ohe.transform(X_test_before_projection_categorial).toarray(),
                               columns=ohe.get_feature_names())
    X_test[one_hot_rep.columns.tolist()] = one_hot_rep
    X_test = X_test.fillna(0)

    # if len(textual_attributes_with_small_domain):
    #     for a_small_domain in textual_attributes_with_small_domain:
    #         X_for_a = X_before_projection.iloc[:, a_small_domain]
    #         one_hot_rep = pd.get_dummies(X_for_a, prefix=str(a_small_domain)).astype(int)
    #         X_test[one_hot_rep.columns.tolist()] = one_hot_rep

    y_test = y_0 + y_1
    solution, validation_labels, generalization_labels = ml_utils.learn_data_transformation_BINARY_CLASSIFICATION(
        X_train,
        y_train,
        X_test,
        y_test,
        True)
    solution = solution.replace('class: 1', 'row removed').replace('class: 0', 'row maintained')
    return solution, validation_labels, generalization_labels


def inspect_full_table_transformation(T_pair_validation, T_pair_generalization):
    X_train = T_pair_validation.T.table
    y_train = T_pair_validation.T_prime.table.iloc[:, :]
    X_test = T_pair_generalization.T.table
    y_test = T_pair_generalization.T_prime.table.iloc[:, :]
    sol = data_transformation_detection(X_train, y_train, X_test, y_test, what_to_explain='full')
    print(sol)
    return sol


def inspect_removed_column(T_pair_validation, T_pair_generalization, removed_column_to_inspect):
    # projected_table = T_pair.T.table.drop([removed_column_to_inspect], axis = 1)
    # print('FIX!')
    start = time.time()
    projected_table_validation = T_pair_validation.T.projected_table
    projected_table_generalization = T_pair_generalization.T.projected_table
    # **Also removing non-matching rows
    projected_column_validation = T_pair_validation.T.table.iloc[:, removed_column_to_inspect]
    projected_column_generalization = T_pair_generalization.T.table.iloc[:, removed_column_to_inspect]
    nan_size_validation = len(projected_column_validation[projected_column_validation == 'nan'])
    nan_size_generalization = len(projected_column_generalization[projected_column_generalization == 'nan'])
    nan_rate_validation = nan_size_validation / len(projected_column_validation)
    nan_rate_generalization = nan_size_generalization / len(projected_column_generalization)
    # print(column_nan_threshold, nan_rate_validation, nan_rate_generalization)
    if nan_rate_validation > column_nan_threshold:
        eval_validation = 1.0
        eval_generalization = 1.0 if nan_rate_generalization > column_nan_threshold else 0.0
        eval_explainabilty = [1, 1, 1]
        run_time = time.time() - start
        return {'CONTAINS_a_lot_of_NANs': [eval_validation, eval_generalization, *eval_explainabilty, run_time]}
    for col in projected_table_validation.columns:
        candidate_col_validation = projected_table_validation[col]  # .to_frame()
        candidate_col_generalization = projected_table_generalization[col]
        # Set-Based
        # overlap = pd.Series(list(set(candidate_col).intersection(set(projected_column))))

        # Overlapping elements
        # if projected_column.dtypes[0] == candidate_col.dtypes[0]:
        #     overlap = projected_column.merge(candidate_col,
        #                                      left_on=projected_column.columns[0],
        #                                      right_on=candidate_col.columns[0],
        #                                      how='inner')
        # else:
        #     overlap = []

        # Explicitly checking mathcing rows
        overlap_validation = len([1 for l_r, r_r in zip(projected_column_validation.tolist(),
                                                        candidate_col_validation.tolist()) if l_r == r_r])
        overlap_generalization = len([1 for l_r, r_r in zip(projected_column_generalization.tolist(),
                                                            candidate_col_generalization.tolist()) if l_r == r_r])
        # print([(l_r, r_r) for l_r, r_r in zip(projected_column.tolist(), candidate_col.tolist())])
        # print(overlap)
        # print(len(candidate_col))
        # print(len(projected_column))
        overlap_ratio_validation = overlap_validation / len(projected_column_validation)
        overlap_ratio_generalization = overlap_generalization / len(projected_column_generalization)
        # print(duplicate_column_overlap_threshold, overlap_ratio_validation, overlap_ratio_generalization)
        if overlap_ratio_validation > duplicate_column_overlap_threshold:
            eval_validation = 1.0
            eval_generalization = 1.0 if overlap_ratio_generalization > duplicate_column_overlap_threshold else 0.0
            eval_explainabilty = [1, 1, 1]
            run_time = time.time() - start
            return {'DUPLICATED (' + str(candidate_col_validation.name) + ')': [eval_validation,
                                                                                eval_generalization,
                                                                                *eval_explainabilty,
                                                                                run_time]}
        return None


def inspect_removed_column_baseline(T_pair_validation,
                                    T_pair_generalization,
                                    removed_column_to_inspect,
                                    extend_for_auto_pipeline=False,
                                    extend_for_plus=False):
    X_train = T_pair_validation.T.table.iloc[:, T_pair_validation.LHCA + [removed_column_to_inspect]]
    X_test = T_pair_generalization.T.table.iloc[:, T_pair_generalization.LHCA + [removed_column_to_inspect]]
    y_train = T_pair_validation.T.projected_table
    y_test = T_pair_generalization.T.projected_table
    solutions = {}
    if extend_for_auto_pipeline:
        sols = data_transformation_detection_column_removal(X_train,
                                                            y_train,
                                                            X_test,
                                                            y_test,
                                                            what_to_explain='auto-pipeline')
        if sols:
            solutions.update(sols)
    elif extend_for_plus:
        sols = data_transformation_detection_column_removal(X_train,
                                                            y_train,
                                                            X_test,
                                                            y_test,
                                                            what_to_explain='foofah_plus')
        if sols:
            solutions.update(sols)
    else:
        sols = data_transformation_detection_column_removal(X_train,
                                                            y_train,
                                                            X_test,
                                                            y_test,
                                                            what_to_explain='foofah')
        if sols:
            solutions.update(sols)
    return solutions


def inspect_removed_rows_baseline(T_pair_validation,
                                  T_pair_generalization,
                                  extend_for_auto_pipeline=False,
                                  extend_for_plus=False):
    X_train = T_pair_validation.T.projected_table
    X_test = T_pair_generalization.T.projected_table
    y_train = T_pair_validation.T_prime.projected_table
    y_test = T_pair_generalization.T_prime.projected_table
    solutions = {}
    if extend_for_auto_pipeline:
        sols = data_transformation_detection_rows_removal(X_train,
                                                          y_train,
                                                          X_test,
                                                          y_test,
                                                          what_to_explain='auto-pipeline')
        if sols:
            solutions.update(sols)
    elif extend_for_plus:
        sols = data_transformation_detection_rows_removal(X_train,
                                                          y_train,
                                                          X_test,
                                                          y_test,
                                                          what_to_explain='foofah_plus')
        if sols:
            solutions.update(sols)
    else:
        sols = data_transformation_detection_rows_removal(X_train,
                                                          y_train,
                                                          X_test,
                                                          y_test,
                                                          what_to_explain='foofah')
        if sols:
            solutions.update(sols)
    return solutions


def inspect_added_rows_baseline(T_pair_validation,
                                T_pair_generalization,
                                extend_for_auto_pipeline=False,
                                extend_for_plus=False):
    y_train = T_pair_validation.T_prime.projected_table
    y_test = T_pair_generalization.T_prime.projected_table
    X_train = T_pair_validation.T.projected_table
    X_test = T_pair_generalization.T.projected_table

    solutions = {}
    if extend_for_auto_pipeline:
        sols = data_transformation_detection_rows_addition(X_train,
                                                           y_train,
                                                           X_test,
                                                           y_test,
                                                           what_to_explain='auto-pipeline')
        if sols:
            solutions.update(sols)
    elif extend_for_plus:
        sols = data_transformation_detection_rows_addition(X_train,
                                                           y_train,
                                                           X_test,
                                                           y_test,
                                                           what_to_explain='foofah_plus')
        if sols:
            solutions.update(sols)
    else:
        sols = data_transformation_detection_rows_addition(X_train,
                                                           y_train,
                                                           X_test,
                                                           y_test,
                                                           what_to_explain='foofah')
        if sols:
            solutions.update(sols)
    return solutions


def inspect_added_row(T_pair, added_row_to_inspect):
    start = time.time()
    orig_table = T_pair.T.projected_table
    added_row = T_pair.T_prime.projected_table.loc[added_row_to_inspect]
    expanded_table = orig_table.append(added_row)
    if expanded_table.duplicated().tolist()[-1]:
        sol = 'BOOTSTRAPPED'
        eval_validation = 1.0
        eval_generalization = 1.0
        eval_explainabilty = [1, 1, 1]
        run_time = time.time() - start
        return {sol: [eval_validation, eval_generalization, *eval_explainabilty, run_time]}
    return {}


def compute_generalizability_for_row_addition(T_pair_generalization, addition_ops):
    correct_additions = 0
    if len(T_pair_generalization.RHDr) == 0:
        return 0
    if addition_ops == ['BOOTSTRAPPED']:
        for row_2_b_explained in T_pair_generalization.RHDr:
            if inspect_added_row(T_pair_generalization, row_2_b_explained):
                correct_additions += 1
    return correct_additions / len(T_pair_generalization.RHDr)


def inspect_group_by_candidate(T_pair_validation, T_pair_generalization, attribute_2_b_explained):
    try:
        T_pair_validation.T_prime.table.iloc[:, attribute_2_b_explained].astype(np.float64)
    except:
        return {}
    group_by_candidates = [i for (i, _) in T_pair_validation.sigma_A]
    group_by_candidates_names = {i: T_pair_validation.T.headers[i] for i in group_by_candidates}
    X_train = T_pair_validation.T.table
    # attributes_to_consider = [i for i in T_pair_validation.T_prime.A if i != T_pair_validation.sigma_A[0][1]]
    # y_train = T_pair_validation.T_prime.table.iloc[:, attribute_2_b_explained]
    y_train = T_pair_validation.T_prime.table.iloc[:, :]
    X_test = T_pair_generalization.T.table
    # y_test = T_pair_generalization.T_prime.table.iloc[:, attribute_2_b_explained]
    y_test = T_pair_generalization.T_prime.table.iloc[:, :]
    sols = {}
    for group_by_candidate in ml_utils.powerset(group_by_candidates):
        if not len(group_by_candidate):
            continue
        names = [group_by_candidates_names[i] for i in group_by_candidate]
        y_train_new = y_train.sort_values(names).iloc[:, attribute_2_b_explained]
        y_test_new = y_test.sort_values(names).iloc[:, attribute_2_b_explained]
        sol = ml_utils.learn_data_transformation_REGRESSION_projected_table(X_train,
                                                                            y_train_new,
                                                                            X_test,
                                                                            y_test_new,
                                                                            group_by_candidate)
        if sol:
            sols.update(sol)
        # print(sol)
    return sols
