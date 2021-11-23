import random
from abc import ABC, abstractmethod
from itertools import islice
from typing import List

import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.dtypes import int32

from utils.model_utils import convert_to_list_of_lists
from data_generators.data_classes import RowSlicer
from data_generators.tokenizer import ConceptTokenizer


def validate_columns_decorator(function):
    """
    A decorator to validate whether the parameter rows passed to LearningObjective.process_batch
    contain the required columns. It raises AttributeError if any of the required columns is
    missing from the rows

    :param function:
    :return:
    """

    def wrapper(self, rows: List[RowSlicer], *args, **kwargs):
        required_columns = self.get_required_columns()
        for row_slicer in rows:
            for column in required_columns:
                if not hasattr(row_slicer.row, column):
                    raise AttributeError(
                        f'The required column {column} is missing for {self}')
            break

        return function(self, rows, *args, **kwargs)

    return wrapper


def post_pad_pre_truncate(inputs, pad_value, max_seq_len, d_type='int32'):
    """
    Post _pad and pre-truncate the sequence

    :param inputs:
    :param pad_value:
    :param max_seq_len:
    :param d_type:
    :return:
    """
    return pad_sequences(np.asarray(inputs),
                         maxlen=max_seq_len, padding='post',
                         value=pad_value, dtype=d_type)


class LearningObjective(ABC):

    @property
    def required_columns(self):
        raise NotImplementedError

    @validate_columns_decorator
    @abstractmethod
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective
        :param rows:
        :return:
        """
        pass

    @abstractmethod
    def get_tf_dataset_schema(self):
        """
        Get the schema for the input and output to the tensorflow Dataset
        :return:
        """
        pass

    @classmethod
    def get_required_columns(cls):
        """
        Get the required columns for this learning objective
        :return:
        """
        return cls.required_columns

    def __str__(self):
        return str(self.__class__.__name__)


class BertFineTuningLearningObjective(LearningObjective):
    required_columns = ['label']

    def get_tf_dataset_schema(self):
        output_dict_schema = {'label': int32}
        return {}, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective
        :param rows:
        :return:
        """
        labels = []
        for row_slicer in rows:
            labels.append(row_slicer.row.label)

        output_dict = {'label': labels}
        return {}, output_dict


class DemographicsLearningObjective(LearningObjective):
    required_columns = ['age', 'gender_concept_id']

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            'age': int32,
            'gender': int32
        }
        return input_dict_schema, {}

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective
        :param rows:
        :return:
        """
        age_input = []
        gender_input = []
        for row_slicer in rows:
            age_input.append(row_slicer.row.age)
            gender_input.append(row_slicer.row.gender_concept_id)

        input_dict = {
            'age': age_input,
            'gender': gender_input
        }

        return input_dict, {}


class ProlongedLengthStayLearningObjective(LearningObjective):
    required_columns = ['prolonged_length_stay']

    def get_tf_dataset_schema(self):
        output_dict_schema = {
            'prolonged_length_stay': int32
        }
        return {}, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective
        :param rows:
        :return:
        """
        prolonged_length_stay_input = []
        for row_slicer in rows:
            prolonged_length_stay_input.append(row_slicer.row.prolonged_length_stay)

        output_dict = {
            'prolonged_length_stay': prolonged_length_stay_input
        }

        return {}, output_dict


class VisitPredictionLearningObjective(LearningObjective):
    required_columns = ['visit_token_ids', 'visit_concept_orders']

    def __init__(self,
                 visit_tokenizer: ConceptTokenizer,
                 max_seq_len: int):
        self._max_seq_len = max_seq_len
        self._visit_tokenizer = visit_tokenizer

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            'masked_visit_concepts': int32,
            'mask_visit': int32
        }
        output_dict_schema = {'visit_predictions': int32}
        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        (output_mask, masked_visit_concepts, visit_concepts) = zip(
            *list(map(self._make_record, rows)))

        unused_token_id = self._visit_tokenizer.get_unused_token_id()

        visit_concepts = post_pad_pre_truncate(visit_concepts,
                                               unused_token_id,
                                               self._max_seq_len)
        masked_visit_concepts = post_pad_pre_truncate(masked_visit_concepts,
                                                      unused_token_id,
                                                      self._max_seq_len)
        visit_mask = (visit_concepts == unused_token_id).astype(int)

        combined_label = np.stack([visit_concepts, output_mask], axis=-1)

        input_dict = {
            'masked_visit_concepts': masked_visit_concepts,
            'mask_visit': visit_mask
        }

        output_dict = {'visit_predictions': combined_label}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield

        :param row_slicer: a namedtuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, left_index, right_index, _ = row_slicer

        iterator = zip(row.visit_concept_orders, row.visit_token_ids)
        (dates, visit_concept_ids) = zip(
            *islice(sorted(iterator, key=lambda tup2: tup2[0]), left_index, right_index))

        masked_visit_concepts, output_mask = self._mask_visit_concepts(
            visit_concept_ids)

        return output_mask, masked_visit_concepts, visit_concept_ids

    def _mask_visit_concepts(self, visit_concepts):
        """
        Any visit has 50% chance to be masked
        :param visit_concepts:
        :return:
        """
        masked_visit_concepts = np.asarray(visit_concepts).copy()
        output_mask = np.zeros((self._max_seq_len,), dtype=int)
        for word_pos in range(0, len(visit_concepts)):
            if random.random() < 0.5:
                output_mask[word_pos] = 1
                masked_visit_concepts[word_pos] = self._visit_tokenizer.get_mask_token_id()
        return masked_visit_concepts, output_mask


class MaskedLanguageModelLearningObjective(LearningObjective):
    required_columns = ['token_ids', 'dates', 'visit_segments', 'ages', 'visit_concept_orders']

    def __init__(self, concept_tokenizer: ConceptTokenizer, max_seq_len: int, is_training: bool):
        self._max_seq_len = max_seq_len
        self._concept_tokenizer = concept_tokenizer
        self._is_training = is_training

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            'masked_concept_ids': int32,
            'concept_ids': int32,
            'mask': int32,
            'time_stamps': int32,
            'visit_segments': int32,
            'ages': int32,
            'visit_concept_orders': int32
        }
        output_dict_schema = {'concept_predictions': int32}
        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):

        (output_mask, masked_concepts, concepts, time_stamps, visit_segments, ages,
         visit_concept_orders) = zip(
            *list(map(self._make_record, rows)))

        unused_token_id = self._concept_tokenizer.get_unused_token_id()

        # The main inputs for bert
        masked_concepts = post_pad_pre_truncate(masked_concepts, unused_token_id, self._max_seq_len)
        concepts = post_pad_pre_truncate(concepts, unused_token_id, self._max_seq_len)
        mask = (concepts == unused_token_id).astype(int)

        # The auxiliary inputs for bert
        visit_segments = post_pad_pre_truncate(visit_segments, 0, self._max_seq_len)
        time_stamps = post_pad_pre_truncate(time_stamps, 0, self._max_seq_len)
        ages = post_pad_pre_truncate(ages, 0, self._max_seq_len)
        visit_concept_orders = post_pad_pre_truncate(visit_concept_orders,
                                                     self._max_seq_len,
                                                     self._max_seq_len)

        input_dict = {'masked_concept_ids': masked_concepts,
                      'concept_ids': concepts,
                      'mask': mask,
                      'time_stamps': time_stamps,
                      'ages': ages,
                      'visit_segments': visit_segments,
                      'visit_concept_orders': visit_concept_orders}

        output_dict = {'concept_predictions': np.stack([concepts, output_mask], axis=-1)}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, left_index, right_index, _ = row_slicer

        sorting_columns = getattr(row, 'orders') if hasattr(row, 'orders') else row.dates

        iterator = zip(map(int, sorting_columns), row.token_ids, row.visit_segments, row.dates,
                       row.ages, row.visit_concept_orders)
        sorted_list = sorted(iterator, key=lambda tup2: (tup2[0], tup2[1]))

        (_, concepts, segments, dates, ages, visit_concept_orders) = zip(
            *list(islice(sorted_list, left_index, right_index)))

        masked_concepts, output_mask = self._mask_concepts(concepts)

        return output_mask, masked_concepts, concepts, dates, segments, ages, visit_concept_orders

    def _mask_concepts(self, concepts):
        """
        Mask out 15% of the concepts
        :param concepts:
        :return:
        """
        masked_concepts = np.asarray(concepts).copy()
        output_mask = np.zeros((self._max_seq_len,), dtype=int)

        if self._is_training:
            for word_pos in range(0, len(concepts)):
                if concepts[word_pos] == self._concept_tokenizer.get_unused_token_id():
                    break
                if random.random() < 0.15:
                    dice = random.random()
                    if dice < 0.8:
                        masked_concepts[word_pos] = self._concept_tokenizer.get_mask_token_id()
                    elif dice < 0.9:
                        masked_concepts[word_pos] = random.randint(
                            self._concept_tokenizer.get_first_token_index(),
                            self._concept_tokenizer.get_last_token_index())
                    # else: 10% of the time we just leave the word as is
                    output_mask[word_pos] = 1

        return masked_concepts, output_mask


class HierarchicalMaskedLanguageModelLearningObjective(LearningObjective):
    required_columns = ['concept_ids', 'dates',
                        'visit_segments', 'ages',
                        'visit_dates', 'visit_masks',
                        'visit_token_ids', 'time_interval_atts']

    def __init__(self, concept_tokenizer: ConceptTokenizer,
                 max_num_of_visits: int,
                 max_num_of_concepts: int,
                 is_training: bool):
        self._concept_tokenizer = concept_tokenizer
        self._max_num_of_visits = max_num_of_visits
        self._max_num_of_concepts = max_num_of_concepts
        self._is_training = is_training

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            'pat_seq': int32,
            'pat_seq_age': int32,
            'pat_seq_time': int32,
            'pat_mask': int32,
            'visit_time_delta_att': int32,
            'visit_mask': int32
        }
        output_dict_schema = {'concept_predictions': int32, 'visit_predictions': int32}
        return input_dict_schema, output_dict_schema

    def _pad(self, x, padded_token):
        return pad_sequences(np.asarray(x), maxlen=self._max_num_of_concepts, padding='post',
                             value=padded_token, dtype='int32')

    def _concept_mask(self, concept_ids):
        return list(map(lambda c: (c == self._concept_tokenizer.get_unused_token_id()).astype(int),
                        concept_ids))

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):

        (
            output_concept_masks, masked_concepts, concepts, dates, ages,
            visit_token_ids, visit_segments, visit_dates, visit_masks, time_interval_atts
        ) = zip(*list(map(self._make_record, rows)))

        unused_token_id = self._concept_tokenizer.get_unused_token_id()

        # The main inputs for bert
        masked_concepts = pd.Series(masked_concepts) \
            .apply(convert_to_list_of_lists) \
            .apply(self._concept_tokenizer.encode) \
            .apply(lambda tokens: self._pad(tokens,
                                            padded_token=unused_token_id))
        concepts = pd.Series(concepts) \
            .apply(convert_to_list_of_lists) \
            .apply(self._concept_tokenizer.encode) \
            .apply(lambda tokens: self._pad(tokens,
                                            padded_token=unused_token_id))

        pat_mask = list(map(self._concept_mask, masked_concepts))

        # The auxiliary inputs for bert
        dates = pd.Series(dates).apply(convert_to_list_of_lists) \
            .apply(lambda time_stamps: self._pad(time_stamps,
                                                 padded_token=0))

        ages = pd.Series(ages).apply(convert_to_list_of_lists) \
            .apply(lambda time_stamps: self._pad(time_stamps,
                                                 padded_token=0))

        input_dict = {'pat_seq': masked_concepts,
                      'pat_mask': pat_mask,
                      'pat_seq_time': dates,
                      'pat_seq_age': ages,
                      'visit_segment': visit_segments,
                      'visit_time_delta_att': time_interval_atts,
                      'visit_mask': visit_masks}
        concepts = np.reshape(np.stack(concepts),
                              (-1, self._max_num_of_concepts * self._max_num_of_visits))
        output_concept_masks = np.reshape(output_concept_masks,
                                          (-1, self._max_num_of_concepts * self._max_num_of_visits))
        output_visit_masks = np.ones_like(visit_token_ids)
        output_dict = {'concept_predictions': np.stack([concepts, output_concept_masks],
                                                       axis=-1),
                       'visit_predictions': np.stack([visit_token_ids, output_visit_masks],
                                                     axis=-1)}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, start_index, end_index, _ = row_slicer

        concepts = row.concept_ids[start_index:end_index]
        dates = row.dates[start_index:end_index]
        ages = row.ages[start_index:end_index]
        visit_segments = row.visit_segments[start_index:end_index]
        visit_dates = row.visit_dates[start_index:end_index]
        visit_masks = row.visit_masks[start_index:end_index]
        visit_token_ids = row.visit_token_ids[start_index:end_index]
        # Skip the first element because there is no time interval for it
        time_interval_atts = row.time_interval_atts[start_index + 1:end_index]

        masked_concepts, output_concept_masks = zip(
            *list(map(self._mask_concepts, row.concept_ids[0:20])))

        return (
            output_concept_masks, masked_concepts, concepts, dates, ages,
            visit_token_ids, visit_segments, visit_dates, visit_masks, time_interval_atts
        )

    def _mask_concepts(self, concepts):
        """
        Mask out 15% of the concepts
        :param concepts:
        :return:
        """
        masked_concepts = np.asarray(concepts).copy()
        output_mask = np.zeros((self._max_num_of_concepts,), dtype=int)

        if self._is_training:
            for word_pos in range(0, len(concepts)):
                if concepts[word_pos] == self._concept_tokenizer.get_unused_token_id():
                    break

                if random.random() < 0.15:
                    dice = random.random()
                    if dice < 0.8:
                        masked_concepts[word_pos] = self._concept_tokenizer.get_mask_token_id()
                    elif dice < 0.9:
                        masked_concepts[word_pos] = random.randint(
                            self._concept_tokenizer.get_first_token_index(),
                            self._concept_tokenizer.get_last_token_index())
                    # else: 10% of the time we just leave the word as is
                    output_mask[word_pos] = 1

        return masked_concepts, output_mask


class TimeAttentionLearningObjective(LearningObjective):
    required_columns = ['token_ids', 'dates']

    def __init__(self, concept_tokenizer: ConceptTokenizer, max_seq_len: int,
                 time_window_size: int):
        super(TimeAttentionLearningObjective, self).__init__()
        self._concept_tokenizer = concept_tokenizer
        self._max_seq_len = max_seq_len
        self._time_window_size = time_window_size

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            'target_concepts': int32,
            'target_time_stamps': int32,
            'context_concepts': int32,
            'context_time_stamps': int32,
            'mask': int32
        }
        output_dict_schema = {'concept_predictions': int32}
        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        (target_concepts, target_dates, context_concepts, context_time_stamps) = zip(
            *list(map(self._make_record, rows)))

        target_concepts = np.asarray(target_concepts)
        target_time_stamps = np.asarray(target_dates)

        context_concepts = post_pad_pre_truncate(context_concepts,
                                                 self._concept_tokenizer.get_unused_token_id(),
                                                 self._max_seq_len)

        context_time_stamps = post_pad_pre_truncate(context_time_stamps, 0, self._max_seq_len)
        mask = (context_concepts == self._concept_tokenizer.get_unused_token_id()).astype(int)

        input_dict = {'target_concepts': target_concepts,
                      'target_time_stamps': target_time_stamps,
                      'context_concepts': context_concepts,
                      'context_time_stamps': context_time_stamps,
                      'mask': mask}

        output_dict = {'concept_predictions': target_concepts}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the time attention data generator to yield

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        target_index = row_slicer.target_index
        start_index = row_slicer.start_index
        end_index = row_slicer.end_index
        concepts = np.asarray(row_slicer.row.token_ids)
        dates = np.asarray(row_slicer.row.dates)

        indexes = np.asarray(list(range(start_index, end_index + 1)))
        indexes = indexes[indexes != target_index]

        return [concepts[target_index]], [dates[target_index]], concepts[indexes], dates[indexes]
