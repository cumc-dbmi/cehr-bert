import os
import re
import datetime
import inspect
import pathlib
import logging
import pickle

from itertools import chain
import pandas as pd
import numpy as np
from pandas import DataFrame as pd_dataframe
from dask.dataframe import DataFrame as dd_dataframe
import tensorflow as tf
from sklearn import metrics
from typing import Dict, Union, Tuple

from tensorflow.data import Dataset
from tensorflow.keras.models import Model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from data_generators.tokenizer import ConceptTokenizer

LOGGER = logging.getLogger(__name__)


def create_folder_if_not_exist(folder, sub_folder_name):
    """
    Create the sub-folder if not exists. Will do not thing if the sub-folder already exists.

    :param folder:
    :param sub_folder_name:
    :return:
    """
    sub_folder = os.path.join(folder, sub_folder_name)
    if not os.path.exists(sub_folder):
        LOGGER.info(f'Create folder: {sub_folder}')
        pathlib.Path(sub_folder).mkdir(parents=True, exist_ok=True)
    return sub_folder


def log_function_decorator(function):
    def wrapper(self, *args, **kwargs):
        function_name = function.__name__
        module_name = inspect.getmodule(function).__name__
        line_no = inspect.getsourcelines(function)[1]

        beginning = datetime.datetime.now()
        logging.getLogger(function.__name__).info(
            f'Started running {module_name}: {function_name} at line {line_no}')
        output = function(self, *args, **kwargs)
        ending = datetime.datetime.now()
        logging.getLogger(function.__name__).info(
            f'Took {ending - beginning} to run {module_name}: {function_name}.')
        return output

    return wrapper


@log_function_decorator
def tokenize_concepts(training_data: Union[pd_dataframe, dd_dataframe],
                      column_name, tokenized_column_name, tokenizer_path,
                      oov_token='0', encode=True, recreate=False):
    """
    Tokenize the concept sequence and save the tokenizer as a pickle file
    :return:
    """
    if not os.path.exists(tokenizer_path) or recreate:
        tokenizer = ConceptTokenizer(oov_token=oov_token)
        tokenizer.fit_on_concept_sequences(training_data[column_name])
    else:
        logging.getLogger(__name__).info(
            f'Loading the existing tokenizer from {tokenizer_path}')
        tokenizer = pickle.load(open(tokenizer_path, 'rb'))

    if encode:
        if isinstance(training_data, dd_dataframe):
            training_data[tokenized_column_name] = training_data[column_name].map_partitions(
                lambda ds: pd.Series(
                    tokenizer.encode(map(lambda t: t[1].tolist(), ds.iteritems()),
                                     is_generator=True),
                    name='concept_ids'), meta='iterable')
        else:
            training_data[column_name] = training_data[column_name].apply(
                lambda concept_ids: concept_ids.tolist() if not isinstance(concept_ids,
                                                                           list) else concept_ids)
            training_data[tokenized_column_name] = tokenizer.encode(training_data[column_name])

    if not os.path.exists(tokenizer_path) or recreate:
        pickle.dump(tokenizer, open(tokenizer_path, 'wb'))
    return tokenizer


def convert_to_list_of_lists(concept_lists):
    return list(map(lambda sub_arrays: sub_arrays.tolist(), concept_lists))


@log_function_decorator
def compute_binary_metrics(model, test_data: Union[Dataset, Tuple[np.ndarray, np.ndarray]],
                           metrics_folder):
    """
    Compute Recall, Precision, F1-score and PR-AUC for the test data

    :param model:
    :param test_data:
    :param metrics_folder:
    :return:
    """

    def run_model():
        if isinstance(test_data, Dataset):
            x = test_data.map(lambda _x, _y: _x)
            y = test_data.map(lambda _x, _y: _y)
            y = list(chain(*y.as_numpy_iterator()))
        elif len(test_data) == 2:
            x, y = test_data
        else:
            raise TypeError('Only numpy array and tensorflow Dataset are supported types.')

        if isinstance(model, Model):
            prob = model.predict(x)
        elif isinstance(model, (LogisticRegression, XGBClassifier, GridSearchCV)):
            prob = model.predict_proba(x)[:, 1]
        else:
            raise TypeError(f'Unknown type for the model {type(model)}')

        return prob, y

    validate_folder(metrics_folder)

    probabilities, labels = run_model()
    precisions, recalls, _ = metrics.precision_recall_curve(labels, np.asarray(probabilities))
    predictions = (np.asarray(probabilities) > 0.5).astype(int)
    recall = metrics.recall_score(labels, predictions, average='binary')
    precision = metrics.precision_score(labels, predictions, average='binary')
    f1_score = metrics.f1_score(labels, predictions, average='binary')
    pr_auc = metrics.auc(recalls, precisions)
    roc_auc = metrics.roc_auc_score(labels, probabilities)

    current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    data_metrics = {'time_stamp': [current_time],
                    'recall': [recall],
                    'precision': [precision],
                    'f1-score': [f1_score],
                    'pr_auc': [pr_auc],
                    'roc_auc': [roc_auc]}

    pd.DataFrame(data_metrics).to_parquet(os.path.join(metrics_folder, f'{current_time}.parquet'))


def save_training_history(history: Dict, history_folder):
    """
    Save the training metrics in the history dictionary as pandas dataframe to the file
    system in parquet format

    :param history:
    :param history_folder:
    :return:
    """

    validate_folder(history_folder)

    current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    history_parquet_file_path = f'{current_time}.parquet'
    data_frame = pd.DataFrame(dict(sorted(history.history.items())))
    data_frame.columns = data_frame.columns.astype(str)
    data_frame.to_parquet(os.path.join(history_folder, history_parquet_file_path))


def validate_folder(folder):
    if not os.path.exists(folder):
        raise FileExistsError(f'{folder} does not exist!')


def create_concept_mask(mask, max_seq_length):
    # mask the third dimension
    concept_mask_1 = tf.tile(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=-1),
                             [1, 1, 1, max_seq_length])
    # mask the fourth dimension
    concept_mask_2 = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
    concept_mask = tf.cast(
        (concept_mask_1 + concept_mask_2) > 0, dtype=tf.int32,
        name=f'{re.sub("[^0-9a-zA-Z]+", "", mask.name)}_mask')
    return concept_mask
