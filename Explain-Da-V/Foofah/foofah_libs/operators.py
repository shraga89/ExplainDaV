import re
import csv
from collections import OrderedDict
import numpy as np
from .prune_rules import contains_empty_col, add_empty_col
import datetime
import random
import string
from textblob import TextBlob
import featuretools as ft
import pandas as pd
import nltk

try:
    nltk.data.find('')
except LookupError:
    nltk.download()
from nltk.corpus import stopwords

try:
    nltk.data.find('corpus/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk import stem

lemmatizer = stem.WordNetLemmatizer()
stopwords_list = set(stopwords.words('english'))

WRAP_1 = True
WRAP_2 = True
WRAP_3 = True
PRUNE_1 = True

delimiters = "[]()-_,!@#$%^&*{}|/;:+`~=' " + '"' + "\\" + "\n"


# delimiters = " " + '"' + "\\" + "\n"


def f_join_char(table, p1, c):
    if p1 > len(table[0]) - 2:
        return None

    new_table = []
    for x in table:
        new_table.append(x[:p1] + [x[p1] + c + x[p1 + 1]] + x[(p1 + 2):])

    return new_table


def f_join(table, p1):
    # Invalid join check
    if p1 > len(table[0]) - 2:
        return None
    new_table = []
    for x in table:
        new_table.append(x[:p1] + [x[p1] + x[p1 + 1]] + x[(p1 + 2):])

    return new_table


# Function of moving column p1 to the end of the table
def f_move_to_end(table, p1):
    # 2d Table
    if isinstance(table[0], list):
        new_table = []
        for x in table:
            new_table.append(x[:p1] + x[p1 + 1:] + [x[p1]])

        return new_table

    # 1d Table
    else:
        return table[:p1] + table[p1 + 1:] + [table[p1]]


# Concatenate every k rows
def f_wrap_every_k_rows(table, k):
    if len(table) % k != 0:
        return None

    a = np.array(table)
    n_row, n_col = a.shape
    b = a.reshape(int(n_row / k), int(n_col * k))
    return b.tolist()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Fill empty cells in column p1 with the values from the upmost cell
def f_fill(table, p1=1):
    # Remove invalid fill if first cell in column p1 is empty
    if not table[0][p1]:
        return None

    last_val = ""
    new_table = []
    for idx, row in enumerate(table):
        if row[p1]:
            last_val = row[p1]
            new_table.append(row)
        else:
            new_row = list(row)
            new_row[p1] = last_val
            new_table.append(new_row)

    return new_table


# Remove rows with empty cells
def f_delete(table, p1):
    new_table = []
    for row in table:
        if row[p1]:
            new_table.append(row)

    return new_table


def f_delete_empty_cols(table):
    new_table = []
    transpose_table = list(map(list, list(zip(*table))))
    for col in transpose_table:
        temp = "".join(col)
        if temp != "":
            new_table.append(col)

    return list(map(list, list(zip(*new_table))))


# Function of moving the last column before column p1 in the table
def f_move_from_end(table, p1):
    # 2d Table
    if isinstance(table[0], list):
        new_table = []
        for x in table:
            new_table.append(x[:p1] + [x[-1]] + x[p1:-1])

        return new_table

    # 1d Table
    else:
        return table[:p1] + [table[-1]] + table[p1:-1]


# Fold all rows starting from p1
# Example: fold on 0
# a1,b1,c1;      a1,b1;
# a2,b2,c2;  =>  a1,c1;
# a3,b3,c3;      a2,b2;
#                a2,c2;
#                a3,b3;
#                a3,c3;
def f_fold(table, p1):
    new_table = []
    rows_per_record = len(table[0]) - p1

    # Handle Exception
    if rows_per_record <= 1 or rows_per_record >= len(table[0]):
        return None

    for row in table:
        for new_row in range(rows_per_record):
            new_record = []
            for col in range(p1):
                new_record.append(row[col])
            new_record.append(row[p1 + new_row])
            new_table.append(new_record)

    return new_table


# Fold all rows after p1
# Example: fold on 0
#   ,header1,header2
# a1,b1     ,c1;          a1,header1,b1;
# a2,b2     ,c2;      =>  a1,header2,c1;
# a3,b3     ,c3;          a2,header1,b2;
#                         a2,header2,c2;
#                         a3,header1,b3;
#                         a3,header2,c3;
def f_fold_header(table, p1):
    new_table = []

    # Remove invalid fold operation
    if len(table) <= 1 or len(table[0]) <= 2 or len(table[0]) - 2 <= p1:
        return None

    for idx, row in enumerate(table):
        if idx > 0:
            for new_row in range(len(table[0]) - p1 - 1):
                new_record = []
                for col in range(p1 + 1):
                    new_record.append(row[col])

                new_record.append(table[0][p1 + 1 + new_row])
                new_record.append(row[p1 + 1 + new_row])
                new_table.append(new_record)

    if PRUNE_1:
        if contains_empty_col(new_table, p1 + 1):
            return None

    return new_table


# Unfold all rows on last column
# Example:
# a1,b1;     a1,b1,c1;
# a1,c1; =>  a2,b2,c2;
# a2,b2;     a3,b3,c3;
# a2,c2;
# a3,b3;
# a3,c3;
def f_unfold(table):
    new_table = []

    if not isinstance(table[0], list):
        return []

    if len(table[0]) < 1:
        return table

    temp = OrderedDict()

    for row in table:
        t = tuple(row[:-1])
        if t in list(temp.keys()):
            temp[t].append(row[-1])
        else:
            temp[t] = [row[-1]]

    max_col = 0
    for key, value in list(temp.items()):
        if len(value) > max_col:
            max_col = len(value)

    for key, value in list(temp.items()):
        temp_row = list(key) + value

        for i in range(max_col - len(value)):
            temp_row.append("")

        new_table.append(temp_row)

    return new_table


# Wrap when column 'col' is the same
def f_wrap(table, col, cur_node=None):
    new_table = []

    if len(table[0]) <= 1:
        return None

    # Remove invalid remove_empty if any empty cell in this column
    if PRUNE_1:
        if contains_empty_col(table, col):
            return None

    keys_list = []
    for row in table:
        keys_list.append(row[col])

    seen = {}
    keys = [seen.setdefault(x, x) for x in keys_list if x not in seen]

    wrap_data = {}
    width = 0
    for key in keys:
        wrap_data[key] = []

    for row in table:
        wrap_data[row[col]] += row[:col] + row[col + 1:]
        if len(wrap_data[row[col]]) > width:
            width = len(wrap_data[row[col]])

    for key in keys:
        new_row = list()
        new_row.append(key)
        new_row += wrap_data[key]
        new_row += ["" for x in range(width - len(wrap_data[key]))]

        new_table.append(new_row)

    return new_table


def f_wrap_one_row(table):
    new_table = [list(np.array(table).ravel())]

    if PRUNE_1:
        if add_empty_col(new_table=new_table, orig_table=table):
            return None

    return new_table


def f_unfold_header(table, key):
    new_table = []
    header_set = []

    # Check invalid unfold_header operation
    if len(table[0]) < 2 or key >= len(table[0]) - 1:
        return None

    if PRUNE_1:
        if contains_empty_col(table, key):
            return None

    for row in table:
        if row[key]:
            header_set.append(row[key])
        else:
            return None

    header_list = list(OrderedDict.fromkeys(header_set))

    # Create header row
    header_row = []
    for i in range(len(table[0]) - 2):
        header_row.append("")

    str_count = 0
    num_count = 0

    for header in header_list:
        header_row.append(header)

        # Header shouldn't be number
        if is_number(header):
            num_count += 1
        else:
            str_count += 1

    if num_count != 0 and str_count != 0:
        return None

    new_table.append(header_row)

    # Create data rows
    if len(table[0]) > 2:
        keys = []
        data = {}

        for row in table:
            new_key = []
            for idx, cell in enumerate(row):
                if idx != key and idx != len(row) - 1:
                    new_key.append(cell)
            keys.append(tuple(new_key))
            data[(tuple(new_key), row[key])] = row[-1]

        new_table_data = []
        key_list = list(OrderedDict.fromkeys(keys))

        for key in key_list:
            new_row = []
            new_row += key

            for header in header_list:
                if (tuple(key), header) in list(data.keys()):
                    new_row.append(data[(tuple(key), header)])
                else:
                    new_row.append("")

            new_table_data.append(new_row)

        new_table += new_table_data

    else:
        new_row_dict = {}
        new_row = []
        for idx, row in enumerate(table):
            if row[key] not in list(new_row_dict.keys()):
                new_row_dict[row[key]] = row[-1]
            # We need to add a new row to new_table
            else:
                for header in header_list:
                    if header in list(new_row_dict.keys()):
                        new_row.append(new_row_dict[header])
                    else:
                        new_row.append("")

                new_table.append(new_row)

                new_row = []
                new_row_dict = {}
                new_row_dict[row[key]] = row[-1]

        for header in header_list:
            if header in list(new_row_dict.keys()):
                new_row.append(new_row_dict[header])
            else:
                new_row.append("")

        new_table.append(new_row)

    return new_table


def f_transpose(table):
    # print('Lets try this function')
    # print(table)
    # print([list(i) for i in zip(*table)])
    # exit()
    return [list(i) for i in zip(*table)]

# def f_swap_columns(table, col1, col2):
#     if not isinstance(table[0], list):
#         return table
#     else:
#         new_table = []
#         for x in table:
#             new_table.append(x[:p1] + x[p1 + 1:] + [x[p1]])
#         return new_table

def _get_col_dtype(col):
    """
    Infer datatype of a pandas column, process only if the column dtype is object.
    input:   col: a pandas Series representing a df column.
    """

    try:
        col_new = col.astype(float)
        return col_new.dtype
    except:
        try:
            col_new = col.astype(int)
            return col_new.dtype
        except:
            return col.dtype
            # return "string"


def f_groupby(table, col, op):
    es = ft.EntitySet(id='main')

    try:
        df = pd.DataFrame(table)
        df.columns = [str(col) for col in df.columns]
        df = df.astype(dict(zip(list(df.columns),
                                [_get_col_dtype(df[col]) for col in df.columns])))
        es.add_dataframe(dataframe_name='main', dataframe=df, make_index=True, index='index')
        es.normalize_dataframe(new_dataframe_name="main_projected",
                               base_dataframe_name="main",
                               index=str(col))
        new_df = ft.dfs(target_dataframe_name="main_projected", entityset=es, agg_primitives=[op])[0].reset_index()
        new_table = new_df.astype(str) .to_json(orient="values")
    except:
        new_table = table
    return new_table


def f_groupby_add_all(table, col, op):
    es = ft.EntitySet(id='main')
    df = pd.DataFrame(table)
    df.columns = [str(col) for col in df.columns]
    df = df.astype(dict(zip(list(df.columns),
                            [_get_col_dtype(df[col]) for col in df.columns])))
    es.add_dataframe(dataframe_name='main', dataframe=df, make_index=True, index='index')
    es.normalize_dataframe(new_dataframe_name="main_projected",
                           base_dataframe_name="main",
                           index=str(col))
    new_df = ft.dfs(target_dataframe_name="main", entityset=es, agg_primitives=[op])[0].reset_index()
    new_table = new_df.astype(str) .to_json(orient="values")
    return new_table


def f_pivot(table, col):
    df = pd.DataFrame(table)
    df.columns = [str(col) for col in df.columns]
    df = df.astype(dict(zip(list(df.columns),
                            [_get_col_dtype(df[col]) for col in df.columns])))
    new_df = df.pivot(index=df.index, columns=col).reset_index()
    new_table = new_df.astype(str) .to_json(orient="values")
    return new_table


def f_unpivot(table, col):
    df = pd.DataFrame(table)
    df.columns = [str(col) for col in df.columns]
    df = df.astype(dict(zip(list(df.columns),
                            [_get_col_dtype(df[col]) for col in df.columns])))
    new_df = df.melt(id_vars=[df.columns[0]], value_vars=[df.columns[col]])
    new_table = new_df.astype(str) .to_json(orient="values")
    return new_table


def f_explode(table, col):
    df = pd.DataFrame(table)
    df.columns = [str(col) for col in df.columns]
    df = df.astype(dict(zip(list(df.columns),
                            [_get_col_dtype(df[col]) for col in df.columns])))
    new_df = df.explode(df.columns[col])
    new_table = new_df.astype(str).to_json(orient="values")
    return new_table

def f_read_csv(raw_data):
    if len(raw_data) == 1 and len(raw_data[0]) == 1:
        input_str = raw_data[0][0]

        rows = input_str.splitlines()

        delimiter_list = ["\t", ",", " "]
        quote_char_list = ["'", '"']

        for delimiter in delimiter_list:
            for quote_char in quote_char_list:
                temp_table = list(csv.reader(rows, delimiter=delimiter, quotechar=quote_char))
                row_len = set()
                for row in temp_table:
                    row_len.add(len(row))

                if len(row_len) == 1 and len(temp_table[0]) > 1:
                    return temp_table

        return raw_data

    else:
        return raw_data


def f_split(table, col, splitter, plus=False):
    # Figure out # of cells to be generated for each row after applying "split" operation
    added_cells = list()
    added_len = 0

    pattern = re.escape(splitter) + "+"

    for row in table:
        temp_cells = row[col].split(splitter)
        if plus:
            temp_cells = re.split(pattern, row[col])

        added_cells.append(temp_cells)
        if len(temp_cells) > added_len:
            added_len = len(temp_cells)

    cloned_cells = list(added_cells)

    for idx, row in enumerate(cloned_cells):
        if len(row) < added_len:
            for i in range(added_len - len(row)):
                added_cells[idx].append("")

    result_table = []

    for idx, row in enumerate(table):
        result_table.append(row[:col] + added_cells[idx] + row[col + 1:])

    if PRUNE_1:
        if add_empty_col(new_table=result_table, orig_table=table):
            return None

    return result_table


def f_split_first(table, col, splitter):
    # Figure out # of cells to be generated for each row after applying "split" operation
    added_cells = list()
    for row in table:
        temp_cells = row[col].split(splitter, 1)
        if len(temp_cells) <= 1:
            temp_cells.append("")
        added_cells.append(temp_cells)

    result_table = []
    for idx, row in enumerate(table):
        result_table.append(row[:col] + added_cells[idx] + row[col + 1:])
    # print(result_table)
    if PRUNE_1:
        if add_empty_col(new_table=result_table, orig_table=table):
            return None

    return result_table


def f_divide_on_comma(table, col):
    new_table = []

    if PRUNE_1:
        if contains_empty_col(table, col):
            return None

    for row in table:
        if "," in row[col]:
            new_table.append(row[:col] + [row[col]] + [""] + row[col + 1:])
        else:
            new_table.append(row[:col] + [""] + [row[col]] + row[col + 1:])

    if PRUNE_1:
        if add_empty_col(new_table=new_table, orig_table=table):
            return None

    return new_table


def f_divide_on_all_alphabets(table, col):
    new_table = []

    if PRUNE_1:
        if contains_empty_col(table, col):
            return None

    for row in table:
        if row[col].isalpha():
            new_table.append(row[:col] + [row[col]] + [""] + row[col + 1:])
        else:
            new_table.append(row[:col] + [""] + [row[col]] + row[col + 1:])

    if PRUNE_1:
        if add_empty_col(new_table=new_table, orig_table=table):
            return None

    return new_table


def f_divide_on_all_digits(table, col):
    new_table = []

    if PRUNE_1:
        if contains_empty_col(table, col):
            return None

    for row in table:
        if row[col].isdigit():
            new_table.append(row[:col] + [row[col]] + [""] + row[col + 1:])
        else:
            new_table.append(row[:col] + [""] + [row[col]] + row[col + 1:])

    if PRUNE_1:
        if add_empty_col(new_table=new_table, orig_table=table):
            return None

    return new_table


def f_divide_on_alphanum(table, col):
    new_table = []

    if PRUNE_1:
        if contains_empty_col(table, col):
            return None

    for row in table:
        if row[col].isalnum():
            new_table.append(row[:col] + [row[col]] + [""] + row[col + 1:])
        else:
            new_table.append(row[:col] + [""] + [row[col]] + row[col + 1:])

    if PRUNE_1:
        if add_empty_col(new_table=new_table, orig_table=table):
            return None

    return new_table


def validate_date(date_text):
    is_date = False
    try:
        datetime.datetime.strptime(date_text, '%m/%d/%Y')
        is_date = True
    except ValueError:
        pass

    return is_date


def f_divide_on_date(table, col):
    new_table = []

    if PRUNE_1:
        if contains_empty_col(table, col):
            return None

    for row in table:
        if validate_date(row[col]):
            new_table.append(row[:col] + [row[col]] + [""] + row[col + 1:])
        else:
            new_table.append(row[:col] + [""] + [row[col]] + row[col + 1:])

    if PRUNE_1:
        if add_empty_col(new_table=new_table, orig_table=table):
            return None

    return new_table


def f_divide_on_dash(table, col):
    new_table = []

    if PRUNE_1:
        if contains_empty_col(table, col):
            return None

    for row in table:
        if "-" in row[col]:
            new_table.append(row[:col] + [row[col]] + [""] + row[col + 1:])
        else:
            new_table.append(row[:col] + [""] + [row[col]] + row[col + 1:])

    if PRUNE_1:
        if add_empty_col(new_table=new_table, orig_table=table):
            return None

    return new_table


# meta_chars = ["\d+", "\\u+", "\l+", "\a+", "[A-Za-z0-9]+", "[A-Za-z0-9]+", "\w+", "[A-Za-z0-9\ ]+"]
meta_chars = ["\d+", r"\\u+", r"\\l+", "\a+", "[A-Za-z0-9]+", "[A-Za-z0-9]+", "\w+", "[A-Za-z0-9\ ]+"]

re_dict = {}
for meta_char in meta_chars:
    pattern = re.compile("^" + meta_char + "$")
    re_dict["^" + meta_char + "$"] = pattern


def find_metachar(s):
    meta_return = []
    for meta_char in meta_chars:
        pattern = re_dict["^" + meta_char + "$"]
        if pattern.match(s):
            meta_return.append(meta_char)

    return meta_return


def infer_regex(string_set):
    regex_set = set()

    str1 = string_set[0]

    alnum_found = False
    alnum_str = ""

    for char in str1:
        if not char.isalnum() and char != ' ':
            if alnum_found:
                meta_chars = find_metachar(alnum_str)
                temp_set = set()

                if not regex_set:
                    for meta in meta_chars:
                        temp_set.add(meta)
                else:
                    for regex in regex_set:
                        for meta in meta_chars:
                            temp_set.add(regex + meta)

                regex_set = temp_set
                alnum_str = ""
                alnum_found = False

            temp_set = set()
            if not regex_set:
                temp_set.add("\\" + char)
            else:
                for regex in regex_set:
                    regex += "\\" + char
                    temp_set.add(regex)

            regex_set = temp_set

        elif alnum_found:
            alnum_str += char

        else:
            alnum_found = True
            alnum_str += char

    # Wrap up
    if alnum_found:
        meta_chars = find_metachar(alnum_str)
        temp_set = set()
        if not regex_set:
            for meta in meta_chars:
                temp_set.add(meta)
        else:
            for regex in regex_set:
                for meta in meta_chars:
                    temp_set.add(regex + meta)

        regex_set = temp_set

    return regex_set


def find_token(string, index, forward=True):
    if index >= len(string):
        return ""

    if forward:
        # Go on until it encouters first alphanumeric
        orig = index

        while index < len(string) and not string[index].isalnum():
            index += 1

        i = 0
        while index + i < len(string) and string[index + i].isalnum():
            i += 1

        return string[orig:index + i]
    else:
        # Go on until it encouters first alphanumeric
        orig = index
        while index - 1 >= 0 and not string[index - 1].isalnum():
            index -= 1

        i = 1
        while index >= i and string[index - i].isalnum():
            i += 1

        return string[index - i + 1:orig]


def f_extract(table, col, regex, prefix="", suffix=""):
    new_table = []

    if prefix or suffix:
        regex = prefix + "(" + regex + ")" + suffix
    else:
        regex = "(" + regex + ")"

    pattern = re.compile(regex)

    for orig_row in table:
        row = list(orig_row)

        m = re.search(pattern, row[col])
        if m:
            found = m.group(1)
            row.insert(col + 1, found)
        else:
            row.insert(col + 1, "")

        new_table.append(row)

    if PRUNE_1:
        if add_empty_col(new_table=new_table, orig_table=table):
            return None

    return new_table


def f_split_w(table, col):
    # Figure out # of cells to be generated for each row after applying "split" operation
    added_cells = list()
    added_len = 0

    for row in table:
        temp_cells = row[col].split()

        added_cells.append(temp_cells)
        if len(temp_cells) > added_len:
            added_len = len(temp_cells)

    cloned_cells = list(added_cells)

    for idx, row in enumerate(cloned_cells):
        if len(row) < added_len:
            for i in range(added_len - len(row)):
                added_cells[idx].append("")

    result_table = []

    for idx, row in enumerate(table):
        result_table.append(row[:col] + added_cells[idx] + row[col + 1:])

    if PRUNE_1:
        if add_empty_col(new_table=result_table, orig_table=table):
            return None

    return result_table


def f_split_tab(table, col):
    # Figure out # of cells to be generated for each row after applying "split" operation
    added_cells = list()
    added_len = 0

    for row in table:
        temp_cells = row[col].split("\t")

        added_cells.append(temp_cells)
        if len(temp_cells) > added_len:
            added_len = len(temp_cells)

    cloned_cells = list(added_cells)

    for idx, row in enumerate(cloned_cells):
        if len(row) < added_len:
            for i in range(added_len - len(row)):
                added_cells[idx].append("")

    result_table = []

    for idx, row in enumerate(table):
        result_table.append(row[:col] + added_cells[idx] + row[col + 1:])

    if PRUNE_1:
        if add_empty_col(new_table=result_table, orig_table=table):
            return None

    return result_table


def f_drop(table, col):
    # print('drop')
    # print(table)
    new_table = []

    # Remove drop if there is only 1 column in the table
    if len(table[0]) == 1:
        return None

    for x in table:
        new_table.append(x[:col] + x[col + 1:])
    # print(new_table)
    return new_table


### Text to numeric ###
def f_count_s(table, col, char):
    # NEW!
    # print('Hi, I am trying this new function!!!')
    result_table = []
    # print(table)
    for row in table:
        # count = str(row[col].count(char))
        # print(row)
        try:
            count = str(len(re.findall(row[col], char)))
        except:
            count = str(row[col].count(char))
        # result_table.append([count, ])
        result_table.append(row[:col + 1] + [count, ] + row[col + 1:])
    return result_table


def f_number_of_words(table, col):
    return f_count_s(table, col, ' ')


def f_number_of_sentences(table, col):
    return f_count_s(table, col, '.')


def f_number_of_rows(table, col):
    return f_count_s(table, col, '\n')


def f_number_of_questions(table, col):
    return f_count_s(table, col, '?')


def f_number_of_emails(table, col):
    return f_count_s(table, col, r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')


def f_number_of_urls(table, col):
    return f_count_s(table, col, 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def f_number_of_ips(table, col):
    return f_count_s(table, col, r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')


def f_number_of_phone_numbers(table, col):
    return f_count_s(table, col,
                     r'[\+\d]?(\d{2,3}[-\.\s]??\d{2,3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')


def f_number_of_punctuations(table, col):
    return f_count_s(table, col, "[" + re.escape(string.punctuation) + "]")


def f_number_of_stopwords(table, col):
    result_table = []
    for row in table:
        count = str(len([word for word in row[col].split() if word in stopwords_list]))
        # result_table.append([count, ])
        result_table.append(row[:col + 1] + [count, ] + row[col + 1:])
    return result_table


### Text to numeric ###
def f_exists_s(table, col, char):
    # NEW!
    # print('Hi, I am trying this new function!!!')
    result_table = []
    for row in table:
        # exists = str(int(char in row[col]))
        exists = str(int(bool(len(re.findall(row[col], char)))))
        # result_table.append([exists, ])
        result_table.append(row[:col + 1] + [exists, ] + row[col + 1:])
    return result_table


def f_contains_multiple_words(table, col):
    return f_exists_s(table, col, ' ')


def f_contains_multiple_sentences(table, col):
    return f_exists_s(table, col, '.')


def f_contains_multiple_rows(table, col):
    return f_exists_s(table, col, '\n')


def f_contains_a_questions(table, col):
    return f_exists_s(table, col, '?')


def f_contains_an_email(table, col):
    return f_exists_s(table, col, r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')


def f_contains_an_url(table, col):
    return f_exists_s(table, col, 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def f_contains_an_ip(table, col):
    return f_exists_s(table, col, r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')


def f_contains_a_phone_number(table, col):
    return f_exists_s(table, col,
                      r'[\+\d]?(\d{2,3}[-\.\s]??\d{2,3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')


def f_contains_a_punctuation(table, col):
    return f_count_s(table, col, "[" + re.escape(string.punctuation) + "]")


def f_contains_a_stopword(table, col):
    result_table = []
    for row in table:
        count = str(bool(len([word for word in row[col].split() if word in stopwords_list])))
        # result_table.append([count, ])
        result_table.append(row[:col + 1] + [count, ] + row[col + 1:])
    return result_table


### Text to numeric ###
def f_len(table, col):
    # NEW!
    # print('Hi, I am trying this new function!!!')
    result_table = []
    for row in table:
        len_str = str(len(row[col]))
        # result_table.append([len_str, ])
        result_table.append(row[:col + 1] + [len_str, ] + row[col + 1:])
    # print(result_table)
    return result_table


seen_values_for_table_col = {}


### Text to numeric ###
def f_is_one_hot(table, col):
    # NEW!
    # print('Hi, I am trying this new function!!!')
    # if (str(table), col) not in seen_values_for_table_col:
    #     seen_values_for_table_col[(str(table), col)] = []
    if col not in seen_values_for_table_col:
        seen_values_for_table_col[col] = []
    # values_in_the_col = list(set([row[col] for row in table if row[col] not in
    #                               seen_values_for_table_col[(str(table), col)]]))
    values_in_the_col = list(set([row[col] for row in table if row[col] not in
                                  seen_values_for_table_col[col]]))
    # if values_in_the_col == ['0', '1']:
    #     return table
    available_cols = list(range(len(table[0])))
    available_cols.remove(col)
    # print('---beginning---')
    # print(values_in_the_col)
    # print(available_cols)
    is_change = False
    while any(val.isnumeric() for val in values_in_the_col) and len(available_cols):
        col = random.sample(available_cols, 1)[0]
        # print(col)
        if col not in seen_values_for_table_col:
            seen_values_for_table_col[col] = []
        values_in_the_col = list(set([row[col] for row in table if row[col] not in
                                      seen_values_for_table_col[col]]))
        is_change = True
        available_cols.remove(col)
    # if is_change:
    #     if col not in seen_values_for_table_col:
    #         seen_values_for_table_col[col] = []
    #     values_in_the_col = list(set([row[col] for row in table if row[col] not in
    #                                   seen_values_for_table_col[col]]))
    # print(values_in_the_col)
    # print('---end---')
    if not len(values_in_the_col) or any(val.isnumeric() for val in values_in_the_col):
        return table
    sampled_val = random.sample(values_in_the_col, 1)[0]
    # seen_values_for_table_col[(str(table), col)] += [sampled_val, ]
    seen_values_for_table_col[col] += [sampled_val, ]
    # print(sampled_val)
    # print(seen_values_for_table_col)
    result_table = []
    for row in table:
        is_value = str(int(row[col] == sampled_val))
        # result_table.append([is_value, ])
        # print(row)
        # print(row[:col+1] + [is_value, ] + row[col + 1:])
        # NOTE: Should I replace or add. Replacing makes it a one shot one we reach the spot. Adding creates a much larger search space
        if len(values_in_the_col):
            result_table.append(row[:col + 1] + [is_value, ] + row[col + 1:])
        else:
            result_table.append(row)
    # print(result_table)
    return result_table


# FULL TABLE OPERATION
def f_normalize(table, col):
    print('To-Be-Implemented')
    return None


# New Transformations (NLP Data Cleaning)

def f_remove_stopwords(table, col):
    result_table = []
    for row in table:
        new_row = ' '.join([word for word in row[col].split() if word not in stopwords_list])
        # result_table.append([count, ])
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    return result_table


def f_remove_numeric(table, col):
    result_table = []
    for row in table:
        new_row = ' '.join([word for word in row[col].split() if not word.isdigit()])
        # result_table.append([count, ])
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    return result_table


def f_remove_punctuation(table, col):
    # print(table)
    result_table = []
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    for row in table:
        new_row = regex.sub('', row[col])
        # result_table.append([count, ])
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    # print(result_table)
    return result_table


def f_remove_url(table, col):
    result_table = []
    regex = re.compile(r'https?://\S+|www\.\S+')
    for row in table:
        new_row = regex.sub('', row[col])
        # result_table.append([count, ])
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    return result_table


def f_remove_html_tags(table, col):
    result_table = []
    regex = re.compile(r'<.*?>')
    for row in table:
        new_row = regex.sub('', row[col])
        # result_table.append([count, ])
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    return result_table


def f_spell_correction(table, col):
    print(table)
    result_table = []
    for row in table:
        new_row = ' '.join([word for word in row[col].split() if not TextBlob(row[col]).correct()])
        # result_table.append([count, ])
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    print(result_table)
    return result_table


def f_lemmatization(table, col):
    # print(table)
    result_table = []
    for row in table:
        new_row = lemmatizer.lemmatize(row[col])
        # new_row = ' '.join([lemmatizer.lemmatize(word) for word in row[col].split()])
        # result_table.append([count, ])
        # if new_row != row[col]:
        #     print(row[col])
        #     print(new_row)
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    # print(result_table)
    return result_table


def f_lower(table, col):
    # print(table)
    result_table = []
    for row in table:
        new_row = row[col].lower()
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    # print(result_table)
    return result_table


def add_ops():
    ops = list()

    ops.append(
        {'name': 'f_fold', 'fxn': lambda x, p1: f_fold(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    if WRAP_1: ops.append(
        {'name': 'f_wrap', 'fxn': lambda x, p1: f_wrap(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    if WRAP_2: ops.append({'name': 'f_wrap_one_row', 'fxn': lambda x: f_wrap_one_row(x), 'params': {}, 'if_col': False,
                           'char': '', 'cost': 1.0,
                           'num_params': 1,
                           })

    ops.append(
        {'name': 'f_fold_header', 'fxn': lambda x, p1: f_fold_header(x, p1), 'params': {}, 'if_col': True, 'char': '',
         'cost': 1.0, 'num_params': 2,
         })
    ops.append(
        {'name': 'f_unfold', 'fxn': lambda x, p1: f_unfold_header(x, p1), 'params': {}, 'if_col': True, 'char': '',
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_drop', 'fxn': lambda x, p1: f_drop(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_split_w', 'fxn': lambda x, p1: f_split_w(x, p1), 'params': {}, 'if_col': True, 'char': 'whitespace',
         'cost': 1.0, 'num_params': 2,
         })
    ops.append(
        {'name': 'f_split_tab', 'fxn': lambda x, p1: f_split_tab(x, p1), 'params': {}, 'if_col': True, 'char': "\t",
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_join_char', 'fxn': lambda x, p1, char="\t": f_join_char(x, p1, char), 'params': {2: '\'\\t\''},
         'if_col': True, 'char': '\t', 'cost': 1.0,
         'num_params': 3, })
    ops.append(
        {'name': 'f_join', 'fxn': lambda x, p1: f_join(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_move_to_end', 'fxn': lambda x, p1: f_move_to_end(x, p1), 'params': {}, 'if_col': True, 'char': '',
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_delete', 'fxn': lambda x, p1: f_delete(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append({'name': 'f_delete_empty_cols', 'fxn': lambda x: f_delete_empty_cols(x), 'params': {}, 'if_col': False,
                'char': '', 'cost': 1.0,
                'num_params': 1, })

    ops.append(
        {'name': 'f_fill', 'fxn': lambda x, p1: f_fill(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    if WRAP_3: ops.append(
        {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 2), 'params': {1: '2'}, 'if_col': False,
         'char': '', 'cost': 1.0, 'num_params': 1,
         })
    if WRAP_3: ops.append(
        {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 3), 'params': {1: '3'}, 'if_col': False,
         'char': '', 'cost': 1.0, 'num_params': 1,
         })
    if WRAP_3: ops.append(
        {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 4), 'params': {1: '4'}, 'if_col': False,
         'char': '', 'cost': 1.0, 'num_params': 1,
         })
    if WRAP_3: ops.append(
        {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 5), 'params': {1: '5'}, 'if_col': False,
         'char': '', 'cost': 1.0, 'num_params': 1,
         })

    ops.append(
        {'name': 'f_divide_on_comma comma', 'fxn': lambda x, p1: f_divide_on_comma(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })
    ops.append(
        {'name': 'f_divide_on_all_digits', 'fxn': lambda x, p1: f_divide_on_all_digits(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })
    ops.append(
        {'name': 'f_divide_on_all_alphabets', 'fxn': lambda x, p1: f_divide_on_all_alphabets(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2, })
    ops.append(
        {'name': 'f_divide_on_alphanum', 'fxn': lambda x, p1: f_divide_on_alphanum(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })
    ops.append({'name': 'f_divide_on_date', 'fxn': lambda x, p1: f_divide_on_date(x, p1), 'params': {}, 'if_col': True,
                'char': '', 'cost': 1.0,
                'num_params': 2, })
    ops.append(
        {'name': 'divide_on f_divide_on_dash', 'fxn': lambda x, p1: f_divide_on_dash(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })

    for s in delimiters:
        ops.append(
            {'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s), 'params': {2: "\'%s\'" % (s)},
             'if_col': True,
             'char': s, 'cost': 1.0, 'num_params': 3,
             })
        ops.append({'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s, True),
                    'params': {2: "\'%s\'" % (s), 3: 'True'}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 4,
                    })
        ops.append({'name': "f_join_char", 'fxn': lambda x, p1, char=s: f_join_char(x, p1, char),
                    'params': {2: "\'%s\'" % (s)}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 3,
                    })
        ops.append(
            {'name': "f_split_first", 'fxn': lambda x, p1, s=s: f_split_first(x, p1, s),
             'params': {2: "\'%s\'" % (s)},
             'if_col': True, 'char': s, 'cost': 1.0,
             'num_params': 3, })

    return ops

def add_ops_auto_pipeline():
    ops = list()

    ops.append(
        {'name': 'f_fold', 'fxn': lambda x, p1: f_fold(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    if WRAP_1: ops.append(
        {'name': 'f_wrap', 'fxn': lambda x, p1: f_wrap(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    if WRAP_2: ops.append({'name': 'f_wrap_one_row', 'fxn': lambda x: f_wrap_one_row(x), 'params': {}, 'if_col': False,
                           'char': '', 'cost': 1.0,
                           'num_params': 1,
                           })
    ops.append(
        {'name': 'f_fold_header', 'fxn': lambda x, p1: f_fold_header(x, p1), 'params': {}, 'if_col': True, 'char': '',
         'cost': 1.0, 'num_params': 2,
         })
    ops.append(
        {'name': 'f_unfold', 'fxn': lambda x, p1: f_unfold_header(x, p1), 'params': {}, 'if_col': True, 'char': '',
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_drop', 'fxn': lambda x, p1: f_drop(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_split_w', 'fxn': lambda x, p1: f_split_w(x, p1), 'params': {}, 'if_col': True, 'char': 'whitespace',
         'cost': 1.0, 'num_params': 2,
         })
    ops.append(
        {'name': 'f_split_tab', 'fxn': lambda x, p1: f_split_tab(x, p1), 'params': {}, 'if_col': True, 'char': "\t",
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_join_char', 'fxn': lambda x, p1, char="\t": f_join_char(x, p1, char), 'params': {2: '\'\\t\''},
         'if_col': True, 'char': '\t', 'cost': 1.0,
         'num_params': 3, })
    ops.append(
        {'name': 'f_join', 'fxn': lambda x, p1: f_join(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_move_to_end', 'fxn': lambda x, p1: f_move_to_end(x, p1), 'params': {}, 'if_col': True, 'char': '',
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_delete', 'fxn': lambda x, p1: f_delete(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append({'name': 'f_delete_empty_cols', 'fxn': lambda x: f_delete_empty_cols(x), 'params': {}, 'if_col': False,
                'char': '', 'cost': 1.0,
                'num_params': 1, })

    ops.append(
        {'name': 'f_fill', 'fxn': lambda x, p1: f_fill(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    if WRAP_3: ops.append(
        {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 2), 'params': {1: '2'}, 'if_col': False,
         'char': '', 'cost': 1.0, 'num_params': 1,
         })
    if WRAP_3: ops.append(
        {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 3), 'params': {1: '3'}, 'if_col': False,
         'char': '', 'cost': 1.0, 'num_params': 1,
         })
    if WRAP_3: ops.append(
        {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 4), 'params': {1: '4'}, 'if_col': False,
         'char': '', 'cost': 1.0, 'num_params': 1,
         })
    if WRAP_3: ops.append(
        {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 5), 'params': {1: '5'}, 'if_col': False,
         'char': '', 'cost': 1.0, 'num_params': 1,
         })

    ops.append(
        {'name': 'f_divide_on_comma comma', 'fxn': lambda x, p1: f_divide_on_comma(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })
    ops.append(
        {'name': 'f_divide_on_all_digits', 'fxn': lambda x, p1: f_divide_on_all_digits(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })
    ops.append(
        {'name': 'f_divide_on_all_alphabets', 'fxn': lambda x, p1: f_divide_on_all_alphabets(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2, })
    ops.append(
        {'name': 'f_divide_on_alphanum', 'fxn': lambda x, p1: f_divide_on_alphanum(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })
    ops.append({'name': 'f_divide_on_date', 'fxn': lambda x, p1: f_divide_on_date(x, p1), 'params': {}, 'if_col': True,
                'char': '', 'cost': 1.0,
                'num_params': 2, })
    ops.append(
        {'name': 'divide_on f_divide_on_dash', 'fxn': lambda x, p1: f_divide_on_dash(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })

    ops.append({'name': 'f_transpose', 'fxn': lambda x: f_transpose(x), 'params': {}, 'if_col': False, 'char': '',
                'cost': 1.0, 'num_params': 1,
                })

    ops.append({'name': 'f_pivot', 'fxn': lambda x, c: f_pivot(x, c), 'params': {},
                'if_col': True,
                'char': '',
                'cost': 1.0, 'num_params': 2,
                })
    ops.append({'name': 'f_unpivot', 'fxn': lambda x, c: f_unpivot(x, c), 'params': {},
                'if_col': True,
                'char': '',
                'cost': 1.0, 'num_params': 2,
                })
    ops.append({'name': 'f_explode', 'fxn': lambda x, c: f_explode(x, c), 'params': {},
                'if_col': True,
                'char': '',
                'cost': 1.0, 'num_params': 2,
                })
    for o in ['min', 'max', 'mean', 'sum', 'std']:
        ops.append({'name': 'f_groupby', 'fxn': lambda x, c, o=o: f_groupby(x, c, o), 'params': {}, 'if_col': True,
                    'char': '',
                    'cost': 1.0, 'num_params': 3,
                    })
        ops.append({'name': 'f_groupby_add_all', 'fxn': lambda x, c, o=o: f_groupby_add_all(x, c, o), 'params': {},
                    'if_col': True,
                    'char': '',
                    'cost': 1.0, 'num_params': 3,
                    })
    # ops.append(
    #     {'name': 'f_swap_columns', 'fxn': lambda x, p1, p2: f_swap_columns(x, p1, p2), 'params': {}, 'if_col': True, 'char': '',
    #      'cost': 1.0, 'num_params': 3,
    #      })


    for s in delimiters:
        ops.append(
            {'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s), 'params': {2: "\'%s\'" % (s)},
             'if_col': True,
             'char': s, 'cost': 1.0, 'num_params': 3,
             })
        ops.append({'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s, True),
                    'params': {2: "\'%s\'" % (s), 3: 'True'}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 4,
                    })
        ops.append({'name': "f_join_char", 'fxn': lambda x, p1, char=s: f_join_char(x, p1, char),
                    'params': {2: "\'%s\'" % (s)}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 3,
                    })
        ops.append(
            {'name': "f_split_first", 'fxn': lambda x, p1, s=s: f_split_first(x, p1, s),
             'params': {2: "\'%s\'" % (s)},
             'if_col': True, 'char': s, 'cost': 1.0,
             'num_params': 3, })

    return ops

def add_ops_full():
    ops = list()
    # ops.append(
    #     {'name': 'f_wrap', 'fxn': lambda x, p1: f_wrap(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
    #      'num_params': 2,
    #      })
    ops.append({'name': 'f_transpose', 'fxn': lambda x: f_transpose(x), 'params': {}, 'if_col': False, 'char': '',
                'cost': 1.0, 'num_params': 1,
                })
    ops.append({'name': 'f_groupby', 'fxn': lambda x, c, o='min': f_groupby(x, c, o), 'params': {}, 'if_col': True,
                'char': '',
                'cost': 1.0, 'num_params': 3,
                })
    # ops.append(
    #     {'name': 'f_swap_columns', 'fxn': lambda x, p1, p2: f_swap_columns(x, p1, p2), 'params': {}, 'if_col': True, 'char': '',
    #      'cost': 1.0, 'num_params': 3,
    #      })

    return ops

def add_ops_row():
    ops = list()

    ops.append(
        {'name': 'f_fold', 'fxn': lambda x, p1: f_fold(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    if WRAP_1: ops.append(
        {'name': 'f_wrap', 'fxn': lambda x, p1: f_wrap(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    if WRAP_2: ops.append({'name': 'f_wrap_one_row', 'fxn': lambda x: f_wrap_one_row(x), 'params': {}, 'if_col': False,
                           'char': '', 'cost': 1.0,
                           'num_params': 1,
                           })
    ops.append(
        {'name': 'f_drop', 'fxn': lambda x, p1: f_drop(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_split_w', 'fxn': lambda x, p1: f_split_w(x, p1), 'params': {}, 'if_col': True, 'char': 'whitespace',
         'cost': 1.0, 'num_params': 2,
         })
    ops.append(
        {'name': 'f_split_tab', 'fxn': lambda x, p1: f_split_tab(x, p1), 'params': {}, 'if_col': True, 'char': "\t",
         'cost': 1.0, 'num_params': 2,
         })
    #
    ops.append(
        {'name': 'f_join_char', 'fxn': lambda x, p1, char="\t": f_join_char(x, p1, char), 'params': {2: '\'\\t\''},
         'if_col': True, 'char': '\t', 'cost': 1.0,
         'num_params': 3, })
    ops.append(
        {'name': 'f_join', 'fxn': lambda x, p1: f_join(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })
    #
    # ops.append(
    #     {'name': 'f_move_to_end', 'fxn': lambda x, p1: f_move_to_end(x, p1), 'params': {}, 'if_col': True, 'char': '',
    #      'cost': 1.0, 'num_params': 2,
    #      })
    #
    ops.append(
        {'name': 'f_delete', 'fxn': lambda x, p1: f_delete(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })
    #
    ops.append({'name': 'f_delete_empty_cols', 'fxn': lambda x: f_delete_empty_cols(x), 'params': {}, 'if_col': False,
                'char': '', 'cost': 1.0,
                'num_params': 1, })
    #
    ops.append(
        {'name': 'f_fill', 'fxn': lambda x, p1: f_fill(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })
    #
    # if WRAP_3: ops.append(
    #     {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 2), 'params': {1: '2'}, 'if_col': False,
    #      'char': '', 'cost': 1.0, 'num_params': 1,
    #      })
    # if WRAP_3: ops.append(
    #     {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 3), 'params': {1: '3'}, 'if_col': False,
    #      'char': '', 'cost': 1.0, 'num_params': 1,
    #      })
    # if WRAP_3: ops.append(
    #     {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 4), 'params': {1: '4'}, 'if_col': False,
    #      'char': '', 'cost': 1.0, 'num_params': 1,
    #      })
    # if WRAP_3: ops.append(
    #     {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 5), 'params': {1: '5'}, 'if_col': False,
    #      'char': '', 'cost': 1.0, 'num_params': 1,
    #      })
    #
    ops.append(
        {'name': 'f_divide_on_comma comma', 'fxn': lambda x, p1: f_divide_on_comma(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })
    # ops.append(
    #     {'name': 'f_divide_on_all_digits', 'fxn': lambda x, p1: f_divide_on_all_digits(x, p1), 'params': {},
    #      'if_col': True,
    #      'char': '', 'cost': 1.0,
    #      'num_params': 2, })
    # ops.append(
    #     {'name': 'f_divide_on_all_alphabets', 'fxn': lambda x, p1: f_divide_on_all_alphabets(x, p1), 'params': {},
    #      'if_col': True, 'char': '', 'cost': 1.0,
    #      'num_params': 2, })
    ops.append(
        {'name': 'f_divide_on_alphanum', 'fxn': lambda x, p1: f_divide_on_alphanum(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })
    ops.append({'name': 'f_divide_on_date', 'fxn': lambda x, p1: f_divide_on_date(x, p1), 'params': {}, 'if_col': True,
                'char': '', 'cost': 1.0,
                'num_params': 2, })
    ops.append(
        {'name': 'divide_on f_divide_on_dash', 'fxn': lambda x, p1: f_divide_on_dash(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })

    ops.append({'name': 'f_transpose', 'fxn': lambda x: f_transpose(x), 'params': {}, 'if_col': False, 'char': '',
                'cost': 1.0, 'num_params': 1,
                })
    for s in delimiters:
        ops.append(
            {'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s), 'params': {2: "\'%s\'" % (s)},
             'if_col': True,
             'char': s, 'cost': 1.0, 'num_params': 3,
             })
        ops.append({'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s, True),
                    'params': {2: "\'%s\'" % (s), 3: 'True'}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 4,
                    })
        ops.append({'name': "f_join_char", 'fxn': lambda x, p1, char=s: f_join_char(x, p1, char),
                    'params': {2: "\'%s\'" % (s)}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 3,
                    })
        ops.append(
            {'name': "f_split_first", 'fxn': lambda x, p1, s=s: f_split_first(x, p1, s),
             'params': {2: "\'%s\'" % (s)},
             'if_col': True, 'char': s, 'cost': 1.0,
             'num_params': 3, })

    return ops


def add_ops_column():
    ops = list()

    # ops.append(
    #     {'name': 'f_fold', 'fxn': lambda x, p1: f_fold(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
    #      'num_params': 2,
    #      })

    # if WRAP_1: ops.append(
    #     {'name': 'f_wrap', 'fxn': lambda x, p1: f_wrap(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
    #      'num_params': 2,
    #      })

    # if WRAP_2: ops.append({'name': 'f_wrap_one_row', 'fxn': lambda x: f_wrap_one_row(x), 'params': {}, 'if_col': False,
    #                        'char': '', 'cost': 1.0,
    #                        'num_params': 1,
    #                        })
    #

    ops.append(
        {'name': 'f_remove_stopwords', 'fxn': lambda x, p1: f_remove_stopwords(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_remove_punctuation', 'fxn': lambda x, p1: f_remove_punctuation(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_remove_numeric', 'fxn': lambda x, p1: f_remove_numeric(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_remove_url', 'fxn': lambda x, p1: f_remove_url(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_remove_html_tags', 'fxn': lambda x, p1: f_remove_html_tags(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    # ops.append(
    #     {'name': 'f_spell_correction', 'fxn': lambda x, p1: f_spell_correction(x, p1), 'params': {}, 'if_col': True,
    #      'char': '', 'cost': 1.0, 'num_params': 2,
    #      })

    ops.append(
        {'name': 'f_lemmatization', 'fxn': lambda x, p1: f_lemmatization(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_lower', 'fxn': lambda x, p1: f_lower(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_unfold', 'fxn': lambda x, p1: f_unfold_header(x, p1), 'params': {}, 'if_col': True, 'char': '',
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_drop', 'fxn': lambda x, p1: f_drop(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 0.5,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_split_w', 'fxn': lambda x, p1: f_split_w(x, p1), 'params': {}, 'if_col': True, 'char': 'whitespace',
         'cost': 1.0, 'num_params': 2,
         })
    ops.append(
        {'name': 'f_split_tab', 'fxn': lambda x, p1: f_split_tab(x, p1), 'params': {}, 'if_col': True, 'char': "\t",
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_join_char', 'fxn': lambda x, p1, char="\t": f_join_char(x, p1, char), 'params': {2: '\'\\t\''},
         'if_col': True, 'char': '\t', 'cost': 1.0,
         'num_params': 3, })
    ops.append(
        {'name': 'f_join', 'fxn': lambda x, p1: f_join(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    # ops.append(
    #     {'name': 'f_move_to_end', 'fxn': lambda x, p1: f_move_to_end(x, p1), 'params': {}, 'if_col': True, 'char': '',
    #      'cost': 1.0, 'num_params': 2,
    #      })

    ops.append(
        {'name': 'f_delete', 'fxn': lambda x, p1: f_delete(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    # ops.append({'name': 'f_delete_empty_cols', 'fxn': lambda x: f_delete_empty_cols(x), 'params': {}, 'if_col': False,
    #             'char': '', 'cost': 1.0,
    #             'num_params': 1, })

    ops.append(
        {'name': 'f_fill', 'fxn': lambda x, p1: f_fill(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    # if WRAP_3: ops.append(
    #     {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 2), 'params': {1: '2'}, 'if_col': False,
    #      'char': '', 'cost': 1.0, 'num_params': 1,
    #      })
    # if WRAP_3: ops.append(
    #     {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 3), 'params': {1: '3'}, 'if_col': False,
    #      'char': '', 'cost': 1.0, 'num_params': 1,
    #      })
    # if WRAP_3: ops.append(
    #     {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 4), 'params': {1: '4'}, 'if_col': False,
    #      'char': '', 'cost': 1.0, 'num_params': 1,
    #      })
    # if WRAP_3: ops.append(
    #     {'name': 'f_wrap_every_k_rows', 'fxn': lambda x: f_wrap_every_k_rows(x, 5), 'params': {1: '5'}, 'if_col': False,
    #      'char': '', 'cost': 1.0, 'num_params': 1,
    #      })

    # ops.append(
    #     {'name': 'f_divide_on_comma comma', 'fxn': lambda x, p1: f_divide_on_comma(x, p1), 'params': {}, 'if_col': True,
    #      'char': '', 'cost': 1.0,
    #      'num_params': 2, })
    # ops.append(
    #     {'name': 'f_divide_on_all_digits', 'fxn': lambda x, p1: f_divide_on_all_digits(x, p1), 'params': {},
    #      'if_col': True,
    #      'char': '', 'cost': 1.0,
    #      'num_params': 2, })
    # ops.append(
    #     {'name': 'f_divide_on_all_alphabets', 'fxn': lambda x, p1: f_divide_on_all_alphabets(x, p1), 'params': {},
    #      'if_col': True, 'char': '', 'cost': 1.0,
    #      'num_params': 2, })
    # ops.append(
    #     {'name': 'f_divide_on_alphanum', 'fxn': lambda x, p1: f_divide_on_alphanum(x, p1), 'params': {}, 'if_col': True,
    #      'char': '', 'cost': 1.0,
    #      'num_params': 2, })
    # ops.append({'name': 'f_divide_on_date', 'fxn': lambda x, p1: f_divide_on_date(x, p1), 'params': {}, 'if_col': True,
    #             'char': '', 'cost': 1.0,
    #             'num_params': 2, })
    ops.append(
        {'name': 'divide_on f_divide_on_dash', 'fxn': lambda x, p1: f_divide_on_dash(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })

    # ops.append({'name': 'f_transpose', 'fxn': lambda x: f_transpose(x), 'params': {}, 'if_col': False, 'char': '',
    #             'cost': 1.0, 'num_params': 1,
    #             })

    for s in delimiters:
        ops.append(
            {'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s), 'params': {2: "\'%s\'" % (s)},
             'if_col': True,
             'char': s, 'cost': 1.0, 'num_params': 3,
             })
        ops.append({'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s, True),
                    'params': {2: "\'%s\'" % (s), 3: 'True'}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 4,
                    })
        ops.append({'name': "f_join_char", 'fxn': lambda x, p1, char=s: f_join_char(x, p1, char),
                    'params': {2: "\'%s\'" % (s)}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 3,
                    })
        ops.append(
            {'name': "f_split_first", 'fxn': lambda x, p1, s=s: f_split_first(x, p1, s),
             'params': {2: "\'%s\'" % (s)},
             'if_col': True, 'char': s, 'cost': 1.0,
             'num_params': 3, })
        # ops.append(
        #     {'name': 'f_count_s', 'fxn': lambda x, p1, s=s: f_count_s(x, p1, s),
        #      'params': {2: "\'%s\'" % (s)},
        #      'if_col': True, 'char': s, 'cost': 1.0,
        #      'num_params': 3, })
        # ops.append(
        #     {'name': 'f_exists_s', 'fxn': lambda x, p1, s=s: f_exists_s(x, p1, s),
        #      'params': {2: "\'%s\'" % (s)},
        #      'if_col': True, 'char': s, 'cost': 1.0,
        #      'num_params': 3, })

    return ops


def add_ops_column_text_to_numeric():
    ops = list()
    global seen_values_for_table_col
    seen_values_for_table_col = {}
    ops.append(
        {'name': 'f_len', 'fxn': lambda x, p1: f_len(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })
    ops.append(
        {'name': 'f_number_of_words', 'fxn': lambda x, p1: f_number_of_words(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_sentences', 'fxn': lambda x, p1: f_number_of_sentences(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_rows', 'fxn': lambda x, p1: f_number_of_rows(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_questions', 'fxn': lambda x, p1: f_number_of_questions(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_emails', 'fxn': lambda x, p1: f_number_of_emails(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_urls', 'fxn': lambda x, p1: f_number_of_urls(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_ips', 'fxn': lambda x, p1: f_number_of_ips(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_phone_numbers', 'fxn': lambda x, p1: f_number_of_phone_numbers(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_punctuations', 'fxn': lambda x, p1: f_number_of_punctuations(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_stopwords', 'fxn': lambda x, p1: f_number_of_stopwords(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    # for s in delimiters:
    #     ops.append(
    #         {'name': 'f_count_s', 'fxn': lambda x, p1, s=s: f_count_s(x, p1, s),
    #          'params': {2: "\'%s\'" % (s)},
    #          'if_col': True, 'char': s, 'cost': 1.0,
    #          'num_params': 3, })
    #     ops.append(
    #         {'name': 'f_exists_s', 'fxn': lambda x, p1, s=s: f_exists_s(x, p1, s),
    #          'params': {2: "\'%s\'" % (s)},
    #          'if_col': True, 'char': s, 'cost': 1.0,
    #          'num_params': 3, })

    return ops


def add_ops_column_text_to_class():
    ops = list()

    ops.append(
        {'name': 'f_contains_multiple_words', 'fxn': lambda x, p1: f_contains_multiple_words(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_multiple_sentences', 'fxn': lambda x, p1: f_contains_multiple_sentences(x, p1),
         'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_multiple_rows', 'fxn': lambda x, p1: f_contains_multiple_rows(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_a_questions', 'fxn': lambda x, p1: f_contains_a_questions(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_an_email', 'fxn': lambda x, p1: f_contains_an_email(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_an_url', 'fxn': lambda x, p1: f_contains_an_url(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_an_ip', 'fxn': lambda x, p1: f_contains_an_ip(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_a_phone_number', 'fxn': lambda x, p1: f_contains_a_phone_number(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_a_punctuation', 'fxn': lambda x, p1: f_contains_a_punctuation(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_a_stopword', 'fxn': lambda x, p1: f_contains_a_stopword(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    # for s in delimiters:
    #     ops.append(
    #         {'name': 'f_count_s', 'fxn': lambda x, p1, s=s: f_count_s(x, p1, s),
    #          'params': {2: "\'%s\'" % (s)},
    #          'if_col': True, 'char': s, 'cost': 1.0,
    #          'num_params': 3, })
    #     ops.append(
    #         {'name': 'f_exists_s', 'fxn': lambda x, p1, s=s: f_exists_s(x, p1, s),
    #          'params': {2: "\'%s\'" % (s)},
    #          'if_col': True, 'char': s, 'cost': 1.0,
    #          'num_params': 3, })

    return ops


def add_ops_plus():
    ops = list()

    ops.append(
        {'name': 'f_remove_stopwords', 'fxn': lambda x, p1: f_remove_stopwords(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_remove_punctuation', 'fxn': lambda x, p1: f_remove_punctuation(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_remove_numeric', 'fxn': lambda x, p1: f_remove_numeric(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_remove_url', 'fxn': lambda x, p1: f_remove_url(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_remove_html_tags', 'fxn': lambda x, p1: f_remove_html_tags(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_lemmatization', 'fxn': lambda x, p1: f_lemmatization(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_lower', 'fxn': lambda x, p1: f_lower(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_unfold', 'fxn': lambda x, p1: f_unfold_header(x, p1), 'params': {}, 'if_col': True, 'char': '',
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_drop', 'fxn': lambda x, p1: f_drop(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 0.5,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_split_w', 'fxn': lambda x, p1: f_split_w(x, p1), 'params': {}, 'if_col': True, 'char': 'whitespace',
         'cost': 1.0, 'num_params': 2,
         })
    ops.append(
        {'name': 'f_split_tab', 'fxn': lambda x, p1: f_split_tab(x, p1), 'params': {}, 'if_col': True, 'char': "\t",
         'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_join_char', 'fxn': lambda x, p1, char="\t": f_join_char(x, p1, char), 'params': {2: '\'\\t\''},
         'if_col': True, 'char': '\t', 'cost': 1.0,
         'num_params': 3, })
    ops.append(
        {'name': 'f_join', 'fxn': lambda x, p1: f_join(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_delete', 'fxn': lambda x, p1: f_delete(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0, 'num_params': 2,
         })

    ops.append(
        {'name': 'f_fill', 'fxn': lambda x, p1: f_fill(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'divide_on f_divide_on_dash', 'fxn': lambda x, p1: f_divide_on_dash(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2, })

    for s in delimiters:
        ops.append(
            {'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s), 'params': {2: "\'%s\'" % (s)},
             'if_col': True,
             'char': s, 'cost': 1.0, 'num_params': 3,
             })
        ops.append({'name': "f_split", 'fxn': lambda x, p1, s=s: f_split(x, p1, s, True),
                    'params': {2: "\'%s\'" % (s), 3: 'True'}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 4,
                    })
        ops.append({'name': "f_join_char", 'fxn': lambda x, p1, char=s: f_join_char(x, p1, char),
                    'params': {2: "\'%s\'" % (s)}, 'if_col': True, 'char': s, 'cost': 1.0,
                    'num_params': 3,
                    })
        ops.append(
            {'name': "f_split_first", 'fxn': lambda x, p1, s=s: f_split_first(x, p1, s),
             'params': {2: "\'%s\'" % (s)},
             'if_col': True, 'char': s, 'cost': 1.0,
             'num_params': 3, })
        # ops.append(
        #     {'name': 'f_count_s', 'fxn': lambda x, p1, s=s: f_count_s(x, p1, s),
        #      'params': {2: "\'%s\'" % (s)},
        #      'if_col': True, 'char': s, 'cost': 1.0,
        #      'num_params': 3, })
        # ops.append(
        #     {'name': 'f_exists_s', 'fxn': lambda x, p1, s=s: f_exists_s(x, p1, s),
        #      'params': {2: "\'%s\'" % (s)},
        #      'if_col': True, 'char': s, 'cost': 1.0,
        #      'num_params': 3, })

    ops.append(
        {'name': 'f_contains_multiple_words', 'fxn': lambda x, p1: f_contains_multiple_words(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_multiple_sentences', 'fxn': lambda x, p1: f_contains_multiple_sentences(x, p1),
         'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_multiple_rows', 'fxn': lambda x, p1: f_contains_multiple_rows(x, p1), 'params': {},
         'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_a_questions', 'fxn': lambda x, p1: f_contains_a_questions(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_an_email', 'fxn': lambda x, p1: f_contains_an_email(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_an_url', 'fxn': lambda x, p1: f_contains_an_url(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_an_ip', 'fxn': lambda x, p1: f_contains_an_ip(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_a_phone_number', 'fxn': lambda x, p1: f_contains_a_phone_number(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_a_punctuation', 'fxn': lambda x, p1: f_contains_a_punctuation(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_contains_a_stopword', 'fxn': lambda x, p1: f_contains_a_stopword(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })
    global seen_values_for_table_col
    seen_values_for_table_col = {}
    ops.append(
        {'name': 'f_len', 'fxn': lambda x, p1: f_len(x, p1), 'params': {}, 'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })
    ops.append(
        {'name': 'f_number_of_words', 'fxn': lambda x, p1: f_number_of_words(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_sentences', 'fxn': lambda x, p1: f_number_of_sentences(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_rows', 'fxn': lambda x, p1: f_number_of_rows(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_questions', 'fxn': lambda x, p1: f_number_of_questions(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_emails', 'fxn': lambda x, p1: f_number_of_emails(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_urls', 'fxn': lambda x, p1: f_number_of_urls(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_ips', 'fxn': lambda x, p1: f_number_of_ips(x, p1), 'params': {}, 'if_col': True,
         'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_phone_numbers', 'fxn': lambda x, p1: f_number_of_phone_numbers(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_punctuations', 'fxn': lambda x, p1: f_number_of_punctuations(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    ops.append(
        {'name': 'f_number_of_stopwords', 'fxn': lambda x, p1: f_number_of_stopwords(x, p1), 'params': {},
         'if_col': True, 'char': '', 'cost': 1.0,
         'num_params': 2,
         })

    # for s in delimiters:
    #     ops.append(
    #         {'name': 'f_count_s', 'fxn': lambda x, p1, s=s: f_count_s(x, p1, s),
    #          'params': {2: "\'%s\'" % (s)},
    #          'if_col': True, 'char': s, 'cost': 1.0,
    #          'num_params': 3, })
    #     ops.append(
    #         {'name': 'f_exists_s', 'fxn': lambda x, p1, s=s: f_exists_s(x, p1, s),
    #          'params': {2: "\'%s\'" % (s)},
    #          'if_col': True, 'char': s, 'cost': 1.0,
    #          'num_params': 3, })

    return ops


def add_extract(current_table, target_table, cur_node=None, goal_node=None):
    myops = []

    cur_table_cols = []
    tar_table_cols = []

    if cur_node and goal_node:
        cur_table_cols = cur_node.prop_col_data
        tar_table_cols = goal_node.prop_col_data

    else:

        for i in range(len(current_table[0])):
            tempset = set([row[i] for row in current_table])

            # Remove empty string
            if "" in tempset:
                tempset.remove("")

            cur_table_cols.append(list(tempset))

        for i in range(len(target_table[0])):
            tempset = set([row[i] for row in target_table])

            # Remove empty string
            if "" in tempset:
                tempset.remove("")

            tar_table_cols.append(list(tempset))

    for i in range(len(tar_table_cols)):
        strs = tar_table_cols[i]

        for j in range(len(cur_table_cols)):

            if_add = True
            # Check if the strings in column i of the target table can always be found in column j of current table. Also we don't want it to be exactly the same.
            if any((item not in "".join(cur_table_cols[j]) or item in cur_table_cols[j]) for item in tar_table_cols[i]):
                break

            if if_add:

                prefix_candidate = []
                suffix_candidate = []

                for item in strs:
                    pattern = re.compile(item)

                    for cell in cur_table_cols[j]:

                        for m in re.finditer(pattern, cell):
                            start = m.start()
                            end = m.end()

                            prev_token = find_token(cell, start, False)

                            if (prev_token.isdigit() != item.isdigit() and prev_token.isalpha() != item.isalpha()):
                                prefix_candidate.append(re.escape(prev_token))

                            next_token = find_token(cell, end)

                            if (next_token.isdigit() != item.isdigit() and next_token.isalpha() != item.isalpha()):
                                suffix_candidate.append(re.escape(next_token))

                prefix_candidate = set(prefix_candidate)

                suffix_candidate = set(suffix_candidate)

                regexes = infer_regex(strs)

                for regex in regexes:
                    # Extract
                    myops.append(
                        {'name': "f_extract", 'fxn': lambda table, col, regex=regex: f_extract(table, col, regex),
                         'params': {2: "\'%s\'" % (regex)}, 'if_col': True, 'char': regex, 'cost': 1.0,
                         'num_params': 3,
                         'preserving_row_major_order': True})

                    for prefix in prefix_candidate:
                        myops.append({'name': "f_extract",
                                      'fxn': lambda table, col, regex=regex, prefix=prefix: f_extract(table, col, regex,
                                                                                                      prefix=prefix),
                                      'params': {2: "\'%s\'" % (regex), 3: "\'%s\'" % (prefix)}, 'if_col': True,
                                      'char': prefix + "(" + regex + ")", 'cost': 1.0,
                                      'num_params': 4,
                                      'preserving_row_major_order': True})

                    for suffix in suffix_candidate:
                        myops.append({'name': "f_extract",
                                      'fxn': lambda table, col, regex=regex, suffix=suffix: f_extract(table, col, regex,
                                                                                                      suffix=suffix),
                                      'params': {2: "\'%s\'" % (regex), 3: "\'%s\'" % (suffix)}, 'if_col': True,
                                      'char': "(" + regex + ")" + suffix, 'cost': 1.0,
                                      'num_params': 4,
                                      'preserving_row_major_order': True})

    return myops
