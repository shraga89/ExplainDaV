import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import subprocess
from pathlib import Path
import glob
import json
import os
# from pandas_profiling import ProfileReport
from config import *
import utils
from Table import Table
from Table_Pair import Table_Pair
import data_gen
import operator
import warnings
import time
from Baselines.internal_baselines import *
import func_timeout


# from Baselines.SQUARES.reproduce import main_squares_in


def run_function(f, max_wait):
    try:
        return func_timeout.func_timeout(max_wait, f)
    except func_timeout.FunctionTimedOut:
        print('timeout')
        return {}
    except:
        print('other error')
        return {}


def save_as_iterative_alternative(T_pair_validation,
                                  T_pair_generalization,
                                  path,
                                  version_counter,
                                  new_problem_set):
    template = path + '/Ver_{}.csv'
    valid_version_id = template.format(str(version_counter))
    version_counter += 1
    T_pair_validation.T.table.to_csv(valid_version_id, index=False)
    generalization_version_id = template.format(str(version_counter))
    version_counter += 1
    T_pair_generalization.T.table.to_csv(generalization_version_id, index=False)
    for attribute_2_b_explained in T_pair_validation.RHDA[:-1]:
        try:
            df_prime_valid_version = template.format(str(version_counter))
            version_counter += 1
            df = T_pair_validation.T_prime.table.iloc[:, T_pair_validation.RHCA + [attribute_2_b_explained]]
            df.to_csv(df_prime_valid_version)
            df_prime_generalization_version = template.format(str(version_counter))
            version_counter += 1
            df = T_pair_generalization.T_prime.table.iloc[:, T_pair_generalization.RHCA + [attribute_2_b_explained]]
            df.to_csv(df_prime_generalization_version)
            new_problem_set.append({'T_validation': valid_version_id,
                                    'T_prime_validation': df_prime_valid_version,
                                    'T_generalization': generalization_version_id,
                                    'T_prime_generalization': df_prime_generalization_version,
                                    'Setup': ''})
        except:
            continue
    for attribute_2_b_explained in T_pair_validation.LHDA:
        df_prime_valid_version = template.format(str(version_counter))
        version_counter += 1
        df = T_pair_validation.T.table.iloc[:, T_pair_validation.LHCA + [attribute_2_b_explained]]
        df.to_csv(df_prime_valid_version)
        df_prime_generalization_version = template.format(str(version_counter))
        version_counter += 1
        df = T_pair_generalization.T.table.iloc[:, T_pair_generalization.LHCA + [attribute_2_b_explained]]
        df.to_csv(df_prime_generalization_version)
        new_problem_set.append({'T_validation': valid_version_id,
                                'T_prime_validation': df_prime_valid_version,
                                'T_generalization': generalization_version_id,
                                'T_prime_generalization': df_prime_generalization_version,
                                'Setup': ''})
        # For rows:
        df_prime_valid_version = template.format(str(version_counter))
        version_counter += 1
        df = T_pair_validation.T_prime.projected_table
        df.to_csv(df_prime_valid_version)
        df_prime_generalization_version = template.format(str(version_counter))
        version_counter += 1
        df = T_pair_generalization.T_prime.projected_table
        df.to_csv(df_prime_generalization_version)
        new_problem_set.append({'T_validation': valid_version_id,
                                'T_prime_validation': df_prime_valid_version,
                                'T_generalization': generalization_version_id,
                                'T_prime_generalization': df_prime_generalization_version,
                                'Setup': ''})
    return version_counter, new_problem_set


def explain_attributes_RHDA(T_pair_validation,
                            T_pair_generalization,
                            is_record_match=True,
                            use_fd_discovery=True,
                            print_all_solutions=False,
                            run_only_reg=False):
    if not is_record_match:
        return explain_attributes_RHDA_no_record_match(T_pair_validation, T_pair_generalization)
    unsolved_targets = []
    attribute_solutions = pd.DataFrame(columns=['target',
                                                'solution',
                                                'validity',
                                                'generalizability',
                                                'explainability_size',
                                                'explainabilty_repeated_terms',
                                                'explainabilty_cognitive_chunks',
                                                'time_to_generate'])
    for attribute_2_b_explained in T_pair_validation.RHDA[:-1]:
        print('-----', attribute_2_b_explained,
              '(' + str(T_pair_validation.T_prime.get_attributes_names([attribute_2_b_explained])) + ') -----')
        RHS_type = T_pair_validation.T_prime.get_attributes_types([attribute_2_b_explained])[0]
        target_task = utils.get_target_type(RHS_type)
        if run_only_reg and target_task != 'REGRESSION':
            print('skipping (only regression)')
            continue
        solutions = {}
        if use_fd_discovery:
            FDs_LHS = utils.find_fd(T_pair_validation, attribute_2_b_explained)
        else:
            FDs_LHS = [T_pair_validation.RHCA]
        # FDs_LHS = [[0]]### REMOVE!!!
        # RHS_type = T_pair_validation.T_prime.get_attributes_types([attribute_2_b_explained])[0]
        # target_task = utils.get_target_type(RHS_type)
        # print(T_prime.get_feature_like_attributes())
        # 'BINARY_CLASSIFICATION'/'MULTICLASS_CLASSIFICATION'/'REGRESSION'/'TEXTUAL'
        # print(target_task)
        # if target_task in ['BINARY_CLASSIFICATION', 'MULTICLASS_CLASSIFICATION', 'REGRESSION']:
        for attribute_set in FDs_LHS:
            print(attribute_set)
            new_solutions = utils.resolve_for_attribute_set(T_pair_validation,
                                                            T_pair_generalization,
                                                            attribute_set,
                                                            attribute_2_b_explained,
                                                            target_task,
                                                            run_only_reg=run_only_reg)
            solutions.update(new_solutions)
        # print(solutions)
        # No find origin:
        print('-- no origin --')
        resolve_for_attribute_set_lambda = lambda: utils.resolve_for_attribute_set(T_pair_validation,
                                                                                   T_pair_generalization,
                                                                                   T_pair_validation.RHCA,
                                                                                   attribute_2_b_explained,
                                                                                   target_task,
                                                                                   run_only_reg=run_only_reg)
        new_solutions = run_function(resolve_for_attribute_set_lambda, foofah_time_limit + 5)
        if len(new_solutions):
            new_solutions = {'(no_find_origin) ' + k: new_solutions[k] for k in new_solutions}
        else:
            new_solutions = {'(no_find_origin) no solution found!': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        solutions.update(new_solutions)
        # print(solutions)

        # Try only Numeric!:
        print('-- only numeric --')
        numeric_attributes = [a for a, a_type in
                              zip(T_pair_validation.T.A,
                                  T_pair_validation.T.get_attributes_types(T_pair_validation.T.A)) if
                              not pd.api.types.is_string_dtype(a_type)]
        if target_task in ['BINARY_CLASSIFICATION', 'MULTICLASS_CLASSIFICATION', 'REGRESSION']:
            new_solutions = utils.resolve_for_attribute_set(T_pair_validation,
                                                            T_pair_generalization,
                                                            numeric_attributes,
                                                            attribute_2_b_explained,
                                                            target_task,
                                                            run_only_reg=run_only_reg)
            if len(new_solutions):
                new_solutions = {'(all_numeric) ' + k: new_solutions[k] for k in new_solutions}
            else:
                new_solutions = {'(all_numeric) no solution found!': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        else:
            new_solutions = {'(all_numeric) no solution found!': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        solutions.update(new_solutions)
        # print(solutions)
        # Try only Textual!:
        print('-- only textual --')
        textual_attributes = [a for a, a_type in
                              zip(T_pair_validation.T.A,
                                  T_pair_validation.T.get_attributes_types(T_pair_validation.T.A)) if
                              pd.api.types.is_string_dtype(a_type)]
        new_solutions = utils.resolve_for_attribute_set(T_pair_validation,
                                                        T_pair_generalization,
                                                        textual_attributes,
                                                        attribute_2_b_explained,
                                                        target_task,
                                                        run_only_reg=run_only_reg)
        if len(new_solutions):
            new_solutions = {'(all_textual) ' + k: new_solutions[k] for k in new_solutions}
        else:
            new_solutions = {'(all_textual) no solution found!': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        solutions.update(new_solutions)
        # print(solutions)

        solution_found = bool(len(solutions))
        best_sol = None
        new_solutions = None
        if solution_found:
            best_sol, best_sol_eval = max(solutions.items(), key=lambda x: sum(x[1]))
            if sum(best_sol_eval) / len(best_sol_eval) < valid_solution_threshold:
                unsolved_targets.append(attribute_2_b_explained)
            # TODO: priority to numerical solutions? What if I have more than one best solution?
            # TODO: implement and return more than one score
            # print(solutions)
            # print('First meet some quality threshold then use the explainability??')
            # print('Best Explanation Found: ->', best_sol, '<- with score of', best_sol_eval)
        else:
            unsolved_targets.append(attribute_2_b_explained)
        if print_all_solutions:
            print('-----')
            print('We found the following solutions:')
            for sol in solutions:
                print(sol, solutions[sol])
                print()
            print('-----')
        solutions_as_df = pd.DataFrame.from_dict(solutions, orient='index', columns=['validity',
                                                                                     'generalizability',
                                                                                     'explainability_size',
                                                                                     'explainabilty_repeated_terms',
                                                                                     'explainabilty_cognitive_chunks',
                                                                                     'time_to_generate'])
        solutions_as_df = solutions_as_df.reset_index().rename(columns={'index': 'solution'})
        solutions_as_df['target'] = '{0}({1})'.format(str(attribute_2_b_explained), str(
            T_pair_validation.T_prime.get_attributes_names([attribute_2_b_explained])[0]))
        if len(solutions_as_df) == 0:
            solutions_as_df = pd.DataFrame(columns=attribute_solutions.columns)
            solutions_as_df.loc[len(solutions_as_df.index)] = ['{0}({1})'.format(str(attribute_2_b_explained), str(
                T_pair_validation.T_prime.get_attributes_names([attribute_2_b_explained])[0])), ''] + [0.0] * 6
        attribute_solutions = pd.concat([attribute_solutions, solutions_as_df[attribute_solutions.columns]])
    return attribute_solutions
    # print(unsolved_targets)
    # break
    # print(attribute_solutions)


def explain_attributes_RHDA_no_record_match(T_pair_validation, T_pair_generalization):
    attribute_solutions = pd.DataFrame(columns=['target',
                                                'solution',
                                                'validity',
                                                'generalizability',
                                                'explainability_size',
                                                'explainabilty_repeated_terms',
                                                'explainabilty_cognitive_chunks',
                                                'time_to_generate'])
    for attribute_2_b_explained in T_pair_validation.RHDA[:]:
        print('-----', attribute_2_b_explained,
              '(' + str(T_pair_validation.T_prime.get_attributes_names([attribute_2_b_explained])) + ') -----')
        solutions = utils.inspect_group_by_candidate(T_pair_validation, T_pair_generalization, attribute_2_b_explained)
        solutions_as_df = pd.DataFrame.from_dict(solutions, orient='index', columns=['validity',
                                                                                     'generalizability',
                                                                                     'explainability_size',
                                                                                     'explainabilty_repeated_terms',
                                                                                     'explainabilty_cognitive_chunks',
                                                                                     'time_to_generate'])
        solutions_as_df = solutions_as_df.reset_index().rename(columns={'index': 'solution'})
        solutions_as_df['target'] = '{0}({1})'.format(str(attribute_2_b_explained), str(
            T_pair_validation.T_prime.get_attributes_names([attribute_2_b_explained])[0]))
        attribute_solutions = pd.concat([attribute_solutions, solutions_as_df[attribute_solutions.columns]])
    return attribute_solutions


def explain_attributes_LHDA(T_pair_validation, T_pair_generalization, is_record_match):
    if not is_record_match:
        return explain_attributes_LHDA_no_record_match(T_pair_validation, T_pair_generalization)
    attribute_solutions = pd.DataFrame(columns=['target',
                                                'solution',
                                                'validity',
                                                'generalizability',
                                                'explainability_size',
                                                'explainabilty_repeated_terms',
                                                'explainabilty_cognitive_chunks',
                                                'time_to_generate'])
    for attribute_2_b_explained in T_pair_validation.LHDA:
        print('-----', attribute_2_b_explained,
              '(' + str(T_pair_validation.T.get_attributes_names([attribute_2_b_explained])) + ') -----')
        solutions = utils.inspect_removed_column(T_pair_validation, T_pair_generalization, attribute_2_b_explained)
        if solutions:
            solutions_as_df = pd.DataFrame.from_dict(solutions, orient='index', columns=['validity',
                                                                                         'generalizability',
                                                                                         'explainability_size',
                                                                                         'explainabilty_repeated_terms',
                                                                                         'explainabilty_cognitive_chunks',
                                                                                         'time_to_generate'])
            solutions_as_df = solutions_as_df.reset_index().rename(columns={'index': 'solution'})
            solutions_as_df['target'] = '{0}({1})'.format(str(attribute_2_b_explained), str(
                T_pair_validation.T.get_attributes_names([attribute_2_b_explained])[0]))
            attribute_solutions = pd.concat([attribute_solutions, solutions_as_df[attribute_solutions.columns]])
        else:
            start = time.time()
            FDs = utils.find_fd(T_pair_validation, attribute_2_b_explained, 'L')
            # print('Explaining ', T_pair_validation.T.get_attributes_names([attribute_2_b_explained]))
            single_origin_fds = []
            # print(len(T_pair_validation.T.table.iloc[:, attribute_2_b_explained].unique()))
            seen_FDs = FDs.copy()
            while FDs:
                origin = FDs.pop()
                if len(origin) == 1 and origin[0] not in single_origin_fds:
                    single_origin_fds.append(*origin)
                if len(origin) > 1:
                    new_FDs = utils.find_fd(T_pair_validation, attribute_2_b_explained, 'L', origin)
                    if len(new_FDs):
                        for fd in new_FDs:
                            if fd not in seen_FDs:
                                FDs.append(fd)
            origin_sorted, cardinalities = utils.get_cardinality(T_pair_validation.T.table,
                                                                 single_origin_fds + [attribute_2_b_explained])
            # print(cardinalities)
            if origin_sorted:
                found_origin = origin_sorted[0]
                FDs_gen = utils.find_fd(T_pair_validation, attribute_2_b_explained, 'L', [found_origin])
                if FDs_gen:
                    eval_generalization = 1.0
                else:
                    eval_generalization = 0.0
                if cardinalities[attribute_2_b_explained] == cardinalities[found_origin]:
                    exp = '1-1 with ' + '{0}({1})'.format(str(found_origin), str(
                        T_pair_validation.T.get_attributes_names([found_origin])[0]))
                else:
                    exp = 'n-1 with ' + '{0}({1})'.format(str(found_origin), str(
                        T_pair_validation.T.get_attributes_names([found_origin])[0]))
                run_time = time.time() - start
                solutions = {exp: [1.0, eval_generalization, 1.0, 1.0, 1.0, run_time]}
                solutions_as_df = pd.DataFrame.from_dict(solutions, orient='index', columns=['validity',
                                                                                             'generalizability',
                                                                                             'explainability_size',
                                                                                             'explainabilty_repeated_terms',
                                                                                             'explainabilty_cognitive_chunks',
                                                                                             'time_to_generate'])
                solutions_as_df = solutions_as_df.reset_index().rename(columns={'index': 'solution'})
                solutions_as_df['target'] = '{0}({1})'.format(str(attribute_2_b_explained), str(
                    T_pair_validation.T.get_attributes_names([attribute_2_b_explained])[0]))
                attribute_solutions = pd.concat([attribute_solutions, solutions_as_df[attribute_solutions.columns]])
    return attribute_solutions


def explain_attributes_LHDA_no_record_match(T_pair_validation, T_pair_generalization):
    attribute_solutions = pd.DataFrame(columns=['target',
                                                'solution',
                                                'validity',
                                                'generalizability',
                                                'explainability_size',
                                                'explainabilty_repeated_terms',
                                                'explainabilty_cognitive_chunks',
                                                'time_to_generate'])
    for attribute_2_b_explained in T_pair_validation.LHDA:
        print('-----', attribute_2_b_explained,
              '(' + str(T_pair_validation.T.get_attributes_names([attribute_2_b_explained])) + ') -----')
        solutions = {'group_by_transformation': [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]}
        solutions_as_df = pd.DataFrame.from_dict(solutions, orient='index', columns=['validity',
                                                                                     'generalizability',
                                                                                     'explainability_size',
                                                                                     'explainabilty_repeated_terms',
                                                                                     'explainabilty_cognitive_chunks',
                                                                                     'time_to_generate'])
        solutions_as_df = solutions_as_df.reset_index().rename(columns={'index': 'solution'})
        solutions_as_df['target'] = '{0}({1})'.format(str(attribute_2_b_explained), str(
            T_pair_validation.T.get_attributes_names([attribute_2_b_explained])[0]))
        attribute_solutions = pd.concat([attribute_solutions, solutions_as_df[attribute_solutions.columns]])
    return attribute_solutions


def explain_rows_LHDr(T_pair_validation, T_pair_generalization):
    row_solutions = pd.DataFrame(columns=['target',
                                          'solution',
                                          'validity',
                                          'generalizability',
                                          'explainability_size',
                                          'explainabilty_repeated_terms',
                                          'explainabilty_cognitive_chunks',
                                          'time_to_generate'])
    unsolved_rows_validation = []
    unsolved_rows_generalization = []
    solved_rows_validation = {}
    solved_rows_generalization = {}
    if len(T_pair_validation.LHDr) == 0:
        return row_solutions
    for row_2_b_explained in T_pair_validation.LHDr:
        # print('--', str(row_2_b_explained),
        #       '(' + str(T.get_row_by_id(row_2_b_explained)) + ') --')
        row_solution = utils.inspect_row(row_2_b_explained, T_pair_validation)
        # print(row_solution)
        if row_solution == 'UNKNOWN':
            unsolved_rows_validation.append(row_2_b_explained)
        else:
            solved_rows_validation[row_2_b_explained] = row_solution
            # print('-----', str(row_2_b_explained), '-----')
            # print(row_solution)
    op_set = utils.get_recreation_operations(solved_rows_validation)
    for row_2_b_explained in T_pair_generalization.LHDr:
        # print('--', str(row_2_b_explained),
        #       '(' + str(T.get_row_by_id(row_2_b_explained)) + ') --')
        row_solution = utils.inspect_row(row_2_b_explained, T_pair_generalization)
        # print(row_solution)
        if row_solution == 'UNKNOWN':
            unsolved_rows_generalization.append(row_2_b_explained)
        else:
            solved_rows_generalization[row_2_b_explained] = row_solution
    # print('-----', str(unsolved_rows_validation), '-----')
    sol, validation_labels, generalization_labels = utils.inspect_multiple_rows(unsolved_rows_validation,
                                                                                T_pair_validation,
                                                                                unsolved_rows_generalization,
                                                                                T_pair_generalization)
    new_sols = {row_id: sol for row_id in unsolved_rows_validation}
    solved_rows_validation.update(new_sols)
    eval_recreation = utils.evaluate_row_removal(T_pair_validation,
                                                 T_pair_generalization,
                                                 op_set,
                                                 validation_labels,
                                                 generalization_labels)
    solutions_as_df = pd.DataFrame.from_dict(solved_rows_validation, orient='index', columns=['solution'])
    solutions_as_df = solutions_as_df.reset_index().rename(columns={'index': 'target'})
    solutions_as_df[['validity',
                     'generalizability',
                     'explainability_size',
                     'explainabilty_repeated_terms',
                     'explainabilty_cognitive_chunks',
                     'time_to_generate']] = eval_recreation
    row_solutions = pd.concat([row_solutions, solutions_as_df[row_solutions.columns]])
    return row_solutions


def explain_rows_RHDr(T_pair_validation, T_pair_generalization):
    row_solutions = pd.DataFrame(columns=['target',
                                          'solution',
                                          'validity',
                                          'generalizability',
                                          'explainability_size',
                                          'explainabilty_repeated_terms',
                                          'explainabilty_cognitive_chunks',
                                          'time_to_generate'])

    for row_2_b_explained in T_pair_validation.RHDr:
        # print('-----', str(row_2_b_explained), '-----')
        solution = utils.inspect_added_row(T_pair_validation, row_2_b_explained)
        if solution:
            solutions_as_df = pd.DataFrame.from_dict(solution, orient='index', columns=['validity',
                                                                                        'generalizability',
                                                                                        'explainability_size',
                                                                                        'explainabilty_repeated_terms',
                                                                                        'explainabilty_cognitive_chunks',
                                                                                        'time_to_generate'])
            solutions_as_df = solutions_as_df.reset_index().rename(columns={'index': 'solution'})
            solutions_as_df['target'] = str(row_2_b_explained)
            row_solutions = pd.concat([row_solutions, solutions_as_df[row_solutions.columns]])
    addition_ops = row_solutions['solution'].unique()
    new_generalizability = utils.compute_generalizability_for_row_addition(T_pair_generalization, addition_ops)
    row_solutions['generalizability'] = new_generalizability
    return row_solutions


def explain_full(T_pair_validation, T_pair_generalization):
    utils.inspect_full_table_transformation(T_pair_validation, T_pair_generalization)


def main_regular():
    df, df_prime = data_gen.table_pair_generation_wine_quality(LH_table_file)
    T = Table(df)
    T_prime = Table(df_prime)
    T_pair = Table_Pair(T, T_prime)
    utils.find_attribute_match(T_pair)
    T_prime.update_projected_table(T_pair.RHCA)
    utils.find_record_match(T_pair)

    print('Attributes to explain:')
    print('LHDA:', T_pair.LHDA, 'RHDA:', T_pair.RHDA)
    print("-" * 50)
    # RHDA_solutions = explain_attributes_RHDA(T_pair)
    # RHDA_solutions['orientation'] = 'Columns'
    # RHDA_solutions['direction'] = 'Addition'
    # explain_attributes_LHDA(T_pair)

    print("-" * 50)
    # print('Rows to explain:')
    print('LHDr:', T_pair.LHDr, 'RHDr:', T_pair.RHDr)
    # explain_rows_LHDr(T_pair)
    # explain_rows_RHDr(T_pair)


def main_full():
    df, df_prime = data_gen.table_pair_generation_wine_quality_full(LH_table_file)
    T = Table(df)
    T_prime = Table(df_prime)
    T_pair = Table_Pair(T, T_prime)
    utils.find_attribute_match(T_pair)
    T_prime.update_projected_table(T_pair.RHCA)
    utils.find_record_match(T_pair, False)

    # print('Attributes to explain:')
    # print('LHDA:', T_pair.LHDA, 'RHDA:', T_pair.RHDA)
    # print("-" * 50)
    # explain_attributes_RHDA(T_pair)
    # explain_attributes_RHDA(print_all_solutions=True)
    # explain_attributes_RHDA(use_fd_discovery=False)
    # print("-" * 50)
    # print('Rows to explain:')
    # print('LHDr:', T_pair.LHDr, 'RHDr:', T_pair.RHDr)
    # print('Explaining full table transformation:')
    # explain_full(T_pair)


def main_with_problem_sets(use_fd_discovery=False,
                           treat_all_numeric_origin=False,
                           treat_all_textual_origin=False,
                           run_only_reg=False):
    start_time = time.time()
    problem_sets = data_gen.load_problem_sets_generate_table_pairs(problem_sets_file)

    output = pd.DataFrame(columns=['problem_set',
                                   'orientation',
                                   'direction',
                                   'target',
                                   'solution',
                                   'validity',
                                   'generalizability',
                                   'explainability_size',
                                   'explainabilty_repeated_terms',
                                   'explainabilty_cognitive_chunks',
                                   'time_to_generate'])
    for i, problem_set in enumerate(problem_sets):
        print('**** Problem Set {} ****'.format(i))
        print(problem_set['Setup'])
        if i in [52, 63, 90, 91, 92, 93]:
            continue
        # if i != 0:
        #     continue
        # if i < 9:
        #     continue
        T_validation = Table(problem_set['T_validation'])
        T_prime_validation = Table(problem_set['T_prime_validation'])
        T_pair_validation = Table_Pair(T_validation, T_prime_validation)
        is_record_match = utils.find_attribute_match(T_pair_validation)
        if is_record_match:
            T_prime_validation.update_projected_table(T_pair_validation.RHCA)
            T_validation.update_projected_table(T_pair_validation.LHCA)
            utils.find_record_match(T_pair_validation)

        T_generalization = Table(problem_set['T_generalization'])
        T_prime_generalization = Table(problem_set['T_prime_generalization'])
        T_pair_generalization = Table_Pair(T_generalization, T_prime_generalization)
        utils.find_attribute_match(T_pair_generalization)
        if is_record_match:
            T_prime_generalization.update_projected_table(T_pair_generalization.RHCA)
            T_generalization.update_projected_table(T_pair_generalization.LHCA)
            utils.find_record_match(T_pair_generalization)
        # if not is_record_match:
        #     explain_full(T_pair_validation, T_pair_generalization)
        #     continue

        # print('Maybe find attribute match without records match and directly search for group by and pivot?')
        # print(is_record_match)
        if len(T_pair_validation.sigma_A) == 0:
            print('remember that this is very easy to resolve!')
            continue
        if len(T_pair_validation.LHDA) > 100 or len(T_pair_validation.RHDA) > 100:
            print('probably a pivot or transpose')
            continue
        print(T_pair_validation.sigma_A)
        print('Attributes to explain:')
        print('LHDA:', T_pair_validation.LHDA, 'RHDA:', T_pair_validation.RHDA)
        print("-" * 50)
        attribute_addition_solutions = explain_attributes_RHDA(T_pair_validation,
                                                               T_pair_generalization,
                                                               is_record_match,
                                                               run_only_reg=run_only_reg)
        attribute_addition_solutions['orientation'] = 'Columns'
        attribute_addition_solutions['direction'] = 'Addition'
        attribute_addition_solutions['problem_set'] = str(i)
        output = pd.concat([output, attribute_addition_solutions[output.columns]])

        if not run_only_reg:
            attribute_removal_solutions = explain_attributes_LHDA(T_pair_validation,
                                                                  T_pair_generalization,
                                                                  is_record_match)
            attribute_removal_solutions['orientation'] = 'Columns'
            attribute_removal_solutions['direction'] = 'Removal'
            attribute_removal_solutions['problem_set'] = str(i)
            output = pd.concat([output, attribute_removal_solutions[output.columns]])

            if is_record_match:
                print("-" * 50)
                print('Rows to explain:')
                print('LHDr:', T_pair_validation.LHDr, 'RHDr:', T_pair_validation.RHDr)
                rows_removal_solutions = explain_rows_LHDr(T_pair_validation, T_pair_generalization)
                rows_removal_solutions['orientation'] = 'Rows'
                rows_removal_solutions['direction'] = 'Removal'
                rows_removal_solutions['problem_set'] = str(i)
                output = pd.concat([output, rows_removal_solutions[output.columns]])

                rows_addition_solutions = explain_rows_RHDr(T_pair_validation, T_pair_generalization)
                rows_addition_solutions['orientation'] = 'Rows'
                rows_addition_solutions['direction'] = 'Addition'
                rows_addition_solutions['problem_set'] = str(i)
                output = pd.concat([output, rows_addition_solutions[output.columns]])

        # Maybe send to full if no solutions found
        output.to_csv('output/output_{}_explain_da_V_{}.csv'.format(dataset_name, start_time))
        output.to_csv('output/output_explain_da_V_most_recent.csv')
        # if not use_fd_discovery:
        #     output.to_csv('output/output_{}_explain_da_V_nofa_{}.csv'.format(dataset_name, time.time()))
        #     output.to_csv('output/output_explain_da_V_nofd_most_recent.csv')
        # if treat_all_numeric_origin:
        #     output.to_csv('output/output_{}_explain_da_V_allnum_{}.csv'.format(dataset_name, time.time()))
        #     output.to_csv('output/output_explain_da_V_allnum_most_recent.csv')
        #     output.to_csv('output/output_explain_da_V_nofd_most_recent.csv')
        # if treat_all_textual_origin:
        #     output.to_csv('output/output_{}_explain_da_V_alltext_{}.csv'.format(dataset_name, time.time()))
        #     output.to_csv('output/output_explain_da_V_alltext_most_recent.csv')
        # else:
        # output.to_csv('output/output_{}_explain_da_V_{}.csv'.format(dataset_name, time.time()))
        # output.to_csv('output/output_explain_da_V_most_recent.csv')


def main_with_problem_sets_baseline_original(extend_for_auto_pipeline=False,
                                             extend_for_plus=False,
                                             use_fd_discovery=False):
    start_time = time.time()
    problem_sets = data_gen.load_problem_sets_generate_table_pairs(problem_sets_file)

    output = pd.DataFrame(columns=['problem_set',
                                   'orientation',
                                   'direction',
                                   'target',
                                   'solution',
                                   'validity',
                                   'generalizability',
                                   'explainability_size',
                                   'explainabilty_repeated_terms',
                                   'explainabilty_cognitive_chunks',
                                   'time_to_generate'])
    for i, problem_set in enumerate(problem_sets):
        print('**** Problem Set {} ****'.format(i))
        print(problem_set['Setup'])
        if i in [52, 63, 90, 91, 92, 93]:
            continue
        # if extend_for_auto_pipeline and i < 15:
        # continue]
        # if i < 3:
        #     continue
        T_validation = Table(problem_set['T_validation'])
        T_prime_validation = Table(problem_set['T_prime_validation'])
        T_pair_validation = Table_Pair(T_validation, T_prime_validation)
        is_record_match = utils.find_attribute_match(T_pair_validation)
        if is_record_match:
            T_prime_validation.update_projected_table(T_pair_validation.RHCA)
            T_validation.update_projected_table(T_pair_validation.LHCA)
            utils.find_record_match(T_pair_validation)

        T_generalization = Table(problem_set['T_generalization'])
        T_prime_generalization = Table(problem_set['T_prime_generalization'])
        T_pair_generalization = Table_Pair(T_generalization, T_prime_generalization)
        utils.find_attribute_match(T_pair_generalization)
        if is_record_match:
            T_prime_generalization.update_projected_table(T_pair_generalization.RHCA)
            T_generalization.update_projected_table(T_pair_generalization.LHCA)
            utils.find_record_match(T_pair_generalization)
        if len(T_pair_validation.sigma_A) == 0:
            print('remember that this is very easy to resolve!')
            continue
        if len(T_pair_validation.LHDA) > 100 or len(T_pair_validation.RHDA) > 100:
            print('probably a pivot or transpose')
            continue
        print(T_pair_validation.sigma_A)
        print('Attributes to explain:')
        print('LHDA:', T_pair_validation.LHDA, 'RHDA:', T_pair_validation.RHDA)
        print("-" * 50)
        attribute_addition_solutions = explain_attributes_RHDA_baseline_iterative(T_pair_validation,
                                                                                  T_pair_generalization,
                                                                                  is_record_match=is_record_match,
                                                                                  use_fd_discovery=use_fd_discovery,
                                                                                  extend_for_auto_pipeline=extend_for_auto_pipeline,
                                                                                  extend_for_plus=extend_for_plus,
                                                                                  print_all_solutions=False)
        attribute_addition_solutions['orientation'] = 'Columns'
        attribute_addition_solutions['direction'] = 'Addition'
        attribute_addition_solutions['problem_set'] = str(i)
        output = pd.concat([output, attribute_addition_solutions[output.columns]])

        attribute_removal_solutions = explain_attributes_LHDA_baseline_iterative(T_pair_validation,
                                                                                 T_pair_generalization,
                                                                                 is_record_match,
                                                                                 extend_for_auto_pipeline=extend_for_auto_pipeline,
                                                                                 extend_for_plus=extend_for_plus)
        attribute_removal_solutions['orientation'] = 'Columns'
        attribute_removal_solutions['direction'] = 'Removal'
        attribute_removal_solutions['problem_set'] = str(i)
        output = pd.concat([output, attribute_removal_solutions[output.columns]])

        if is_record_match:
            print("-" * 50)
            print('Rows to explain:')
            print('LHDr:', T_pair_validation.LHDr, 'RHDr:', T_pair_validation.RHDr)
            rows_removal_solutions = explain_rows_LHDr_baseline_iterative(T_pair_validation,
                                                                          T_pair_generalization,
                                                                          extend_for_auto_pipeline=extend_for_auto_pipeline,
                                                                          extend_for_plus=extend_for_plus)
            rows_removal_solutions['orientation'] = 'Rows'
            rows_removal_solutions['direction'] = 'Removal'
            rows_removal_solutions['problem_set'] = str(i)
            output = pd.concat([output, rows_removal_solutions[output.columns]])

            rows_addition_solutions = explain_rows_RHDr_baseline_iterative(T_pair_validation, T_pair_generalization)
            rows_addition_solutions['orientation'] = 'Rows'
            rows_addition_solutions['direction'] = 'Addition'
            rows_addition_solutions['problem_set'] = str(i)
            output = pd.concat([output, rows_addition_solutions[output.columns]])

            # Maybe send to full if no solutions found
            if extend_for_plus:
                output.to_csv('output/output_{}_foofahplus_nofd_{}.csv'.format(dataset_name, start_time))
                output.to_csv('output/output_foofahplus_nofd_most_recent.csv')
            elif extend_for_auto_pipeline:
                output.to_csv('output/output_{}_autopipeline_nofd_{}.csv'.format(dataset_name, start_time))
                output.to_csv('output/output_autopipeline_nofd_most_recent.csv')
            else:
                output.to_csv('output/output_{}_foofah_nofd_{}.csv'.format(dataset_name, start_time))
                output.to_csv('output/output_foofah_nofd_most_recent.csv')


def main_save_alternatives():
    problem_sets = data_gen.load_problem_sets_generate_table_pairs(problem_sets_file)
    version_counter = 0
    new_problem_set = []
    path = problem_sets_file.rsplit('/', 1)[0] + '_chanked'
    if not os.path.exists(path):
        os.makedirs(path)
    for i, problem_set in enumerate(problem_sets):
        print('**** Problem Set {} ****'.format(i))
        T_validation = Table(problem_set['T_validation'])
        T_prime_validation = Table(problem_set['T_prime_validation'])
        T_pair_validation = Table_Pair(T_validation, T_prime_validation)
        is_record_match = utils.find_attribute_match(T_pair_validation)
        if is_record_match:
            T_prime_validation.update_projected_table(T_pair_validation.RHCA)
            T_validation.update_projected_table(T_pair_validation.LHCA)
            utils.find_record_match(T_pair_validation)

        T_generalization = Table(problem_set['T_generalization'])
        T_prime_generalization = Table(problem_set['T_prime_generalization'])
        T_pair_generalization = Table_Pair(T_generalization, T_prime_generalization)
        utils.find_attribute_match(T_pair_generalization)
        if is_record_match:
            T_prime_generalization.update_projected_table(T_pair_generalization.RHCA)
            T_generalization.update_projected_table(T_pair_generalization.LHCA)
            utils.find_record_match(T_pair_generalization)

        version_counter, new_problem_set = save_as_iterative_alternative(T_pair_validation,
                                                                         T_pair_generalization,
                                                                         path,
                                                                         version_counter,
                                                                         new_problem_set)
    pd.DataFrame(new_problem_set).to_csv(path + '/problem_sets.csv', index=False)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main_with_problem_sets()
    # main_with_problem_sets(run_only_reg=True)
    # main_with_problem_sets(use_fd_discovery=False)
    # main_with_problem_sets(treat_all_numeric_origin=True)
    # main_with_problem_sets(treat_all_textual_origin=True)
    # print('Foofah')
    # main_with_problem_sets_baseline_original(extend_for_auto_pipeline=False, extend_for_plus=False)
    # print('Auto-pipeline')
    # main_with_problem_sets_baseline_original(extend_for_auto_pipeline=True, extend_for_plus=False)
    # print('Foofah +')
    # main_with_problem_sets_baseline_original(extend_for_auto_pipeline=False, extend_for_plus=True)
    # main_squares()
    # main_regular()
    # main_full()
    # main_save_alternatives()
