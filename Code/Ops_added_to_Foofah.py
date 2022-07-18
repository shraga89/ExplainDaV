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


### Text to numeric ###


def f_count_s(table, col, char):
    result_table = []
    for row in table:
        try:
            count = str(len(re.findall(row[col], char)))
        except:
            count = str(row[col].count(char))
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


def f_len(table, col):
    result_table = []
    for row in table:
        len_str = str(len(row[col]))
        result_table.append(row[:col + 1] + [len_str, ] + row[col + 1:])
    return result_table


### Text to class ###

def f_exists_s(table, col, char):
    result_table = []
    for row in table:
        exists = str(int(bool(len(re.findall(row[col], char)))))
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
        result_table.append(row[:col + 1] + [count, ] + row[col + 1:])
    return result_table


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
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    return result_table


def f_spell_correction(table, col):
    result_table = []
    for row in table:
        new_row = ' '.join([word for word in row[col].split() if not TextBlob(row[col]).correct()])
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    return result_table


def f_lemmatization(table, col):
    result_table = []
    for row in table:
        new_row = lemmatizer.lemmatize(row[col])
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    return result_table


def f_lower(table, col):
    result_table = []
    for row in table:
        new_row = row[col].lower()
        result_table.append(row[:col + 1] + [new_row, ] + row[col + 1:])
    return result_table
