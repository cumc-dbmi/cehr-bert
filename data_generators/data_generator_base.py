import logging
from typing import Set
import inspect

import math
from collections import ChainMap
from pandas import DataFrame
from itertools import chain

from data_generators.learning_objective import *
from data_generators.tokenizer import ConceptTokenizer


def create_indexes_by_time_window(dates, cursor, max_seq_len,
                                  time_window_size):
    """
    Extract the start_index and end_index used for slicing the sequences e.g. concept_ids and dates

    :param dates: a list of time stamps associated with the context
    :param cursor: the current index used as the center for slicing the sequence
    :param max_seq_len: the maximum sequence length
    :param time_window_size: the maximum time window allowed
    :return: start_index and end_index
    """
    seq_len = len(dates)
    half_context_window_size = int(max_seq_len / 2)
    start_index = max(0, cursor - half_context_window_size)
    end_index = min(cursor + half_context_window_size, seq_len)

    half_time_window_size = int(time_window_size / 2)
    context_dates = dates[start_index:end_index]
    time_deltas = context_dates - dates[cursor]
    context_indexes = np.squeeze(
        np.argwhere((time_deltas >= -half_time_window_size)
                    & (time_deltas <= half_time_window_size)),
        axis=-1)

    return np.min(context_indexes).item(), np.max(context_indexes).item()


def get_required_params(clazz: LearningObjective):
    """
    Get required parameters for the learning objective class
    :param clazz:
    :return:
    """
    params = inspect.signature(clazz).parameters
    return [
        dict(name=name, required=param.default is inspect.Parameter.empty)
        for name, param in params.items()
    ]


class AbstractDataGeneratorBase(ABC):
    default_min_num_of_concepts = 2
    default_required_column = 'concept_ids'

    def __init__(self,
                 training_data: DataFrame,
                 batch_size: int,
                 max_seq_len: int,
                 min_num_of_concepts: int,
                 is_random_cursor: bool = False,
                 is_training: bool = True,
                 *args,
                 **kwargs):

        self._training_data = training_data
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._min_num_of_concepts = min_num_of_concepts
        self._is_random_cursor = is_random_cursor
        self._is_training = is_training

        self.get_logger().info(f'batch_size: {batch_size}\n'
                               f'max_seq_len: {max_seq_len}\n'
                               f'min_num_of_concepts: {min_num_of_concepts}\n'
                               f'is_random_cursor: {is_random_cursor}\n'
                               f'is_training: {is_training}\n')

        self._learning_objectives = self._initialize_learning_objectives(
            max_seq_len=max_seq_len, is_training=is_training, **kwargs)
        # validate the required columns in the training data
        self._validate_data_frame_columns()
        self._clean_dataframe()

    @abstractmethod
    def _get_learning_objective_classes(self) -> List[LearningObjective]:
        """
        Initialize a list of LearningObjectives used for generating the input and and output
        :return:
        """
        pass

    def _initialize_learning_objectives(self,
                                        **kwargs) -> List[LearningObjective]:
        """
        Initialize a list of LearningObjectives used for generating the input and and output
        :return:
        """

        def _initialize(learning_objective) -> LearningObjective:
            """
            Initialize one LearningObjective using the provided keyword arguments
            from the parent method

            :param learning_objective:
            :return:
            """
            learning_object_input = dict()
            params = get_required_params(learning_objective)
            for required_param in [
                param['name'] for param in params if param['required']
            ]:
                if required_param in kwargs:
                    learning_object_input[required_param] = kwargs[
                        required_param]
            return learning_objective(**learning_object_input)

        return list(map(_initialize, self._get_learning_objective_classes()))

    def _validate_data_frame_columns(self):
        """
        Validate if the training data has all required columns
        :return:
        """
        dataframe_columns = self._training_data.columns.tolist()
        for required_column in self._get_required_columns():
            if not required_column in dataframe_columns:
                raise ValueError(
                    f'The required column {required_column} does not exist in the training data'
                )

    def _clean_dataframe(self):
        """
        Clean the input data (_training_data) e.g. remove rows whose sequence length is less than
        _minimum_num_of_concepts.

        Overload this method in the subclasses to overwrite the default behavior

        :return:
        """
        self._training_data = self._training_data[
            self._training_data[self.default_required_column].apply(
                lambda token_ids: len(token_ids)) >=
            max(self.default_min_num_of_concepts, self._min_num_of_concepts)]

    def create_batch_generator(self):
        """
        Create the batch generator for tf.dataset.from_generator to use
        :return:
        """

        def filter_lambda(row_slicer):
            """
            Filter out the row_slicer whose concept_ids are less than min_num_of_concepts
            :param row_slicer:
            :return:
            """
            return len(getattr(
                row_slicer.row,
                self.default_required_column)) >= self._min_num_of_concepts

        iterator = self._create_iterator()

        while True:

            rows = list(
                filter(filter_lambda, islice(iterator, self._batch_size)))

            input_dicts = []
            output_dicts = []

            for learning_objective in self._learning_objectives:
                input_dict, output_dict = learning_objective.process_batch(
                    rows)
                input_dicts.append(input_dict)
                output_dicts.append(output_dict)

            yield dict(ChainMap(*input_dicts)), dict(ChainMap(*output_dicts))

    def set_learning_objectives(self,
                                learning_objectives: List[LearningObjective]):
        """
        Overwrite the default learning objectives

        :param learning_objectives:
        :return:
        """
        self._learning_objectives = learning_objectives

    @abstractmethod
    def _create_iterator(self) -> RowSlicer:
        pass

    @abstractmethod
    def estimate_data_size(self):
        pass

    def get_steps_per_epoch(self):
        return self.estimate_data_size() // self._batch_size \
               + self.estimate_data_size() % self._batch_size

    def _get_required_columns(self) -> Set[str]:
        """
        Combine lists of required columns from multiple learning objectives into a unique set of
        required columns

        :return:
        """
        learning_objective_required_columns = list(
            chain(*[
                learning_objective.get_required_columns()
                for learning_objective in self._learning_objectives
            ]))
        return set(learning_objective_required_columns +
                   [self.default_required_column])

    def get_tf_dataset_schema(self):
        """
        Combine the input and output tensorflow data schema from multiple learning objectives
        :return:
        """
        input_dict_schemas = []
        output_dict_schemas = []
        for learning_objective in self._learning_objectives:
            input_dict_schema, output_dict_schema = learning_objective.get_tf_dataset_schema(
            )
            input_dict_schemas.append(input_dict_schema)
            output_dict_schemas.append(output_dict_schema)
        return dict(ChainMap(*input_dict_schemas)), dict(
            ChainMap(*output_dict_schemas))

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)


class BertDataGenerator(AbstractDataGeneratorBase):
    def __init__(self, concept_tokenizer: ConceptTokenizer, *args, **kwargs):
        super(BertDataGenerator,
              self).__init__(concept_tokenizer=concept_tokenizer,
                             *args,
                             **kwargs)
        self._concept_tokenizer = concept_tokenizer

    def _get_learning_objective_classes(self):
        return [MaskedLanguageModelLearningObjective]

    def _create_iterator(self):
        """
        Create an iterator that will iterate forever
        :return:
        """
        while True:
            for row in self._training_data.itertuples():
                seq_length = len(row.token_ids)
                if self._is_training:
                    cursor = random.randint(0, seq_length -
                                            1) if self._is_random_cursor & (
                            seq_length > self._max_seq_len
                    ) else seq_length // 2

                    half_window_size = int(self._max_seq_len / 2)
                    start_index = max(0, cursor - half_window_size)
                    end_index = min(cursor + half_window_size, seq_length)

                    if start_index < end_index:
                        yield RowSlicer(row, start_index, end_index)
                else:
                    yield RowSlicer(row, 0, seq_length)

    def estimate_data_size(self):
        return len(self._training_data)


class BertVisitPredictionDataGenerator(BertDataGenerator):
    def __init__(self, visit_tokenizer: ConceptTokenizer, *args, **kwargs):
        super(BertDataGenerator,
              self).__init__(visit_tokenizer=visit_tokenizer, *args, **kwargs)
        self._visit_tokenizer = visit_tokenizer

    def _get_learning_objective_classes(self):
        return [
            MaskedLanguageModelLearningObjective,
            VisitPredictionLearningObjective
        ]


class HierarchicalBertDataGenerator(AbstractDataGeneratorBase):
    def __init__(self,
                 concept_tokenizer: ConceptTokenizer,
                 max_num_of_visits: int,
                 max_num_of_concepts: int,
                 sliding_window: int = 10,
                 min_num_of_concepts: int = 5,
                 *args,
                 **kwargs):

        max_seq_len = max_num_of_visits * max_num_of_concepts
        super(HierarchicalBertDataGenerator,
              self).__init__(concept_tokenizer=concept_tokenizer,
                             max_num_of_visits=max_num_of_visits,
                             max_num_of_concepts=max_num_of_concepts,
                             max_seq_len=max_seq_len,
                             min_num_of_concepts=min_num_of_concepts,
                             *args,
                             **kwargs)
        self._concept_tokenizer = concept_tokenizer
        self._max_num_of_visits = max_num_of_visits
        self._max_num_of_concepts = max_num_of_concepts
        self._sliding_window = sliding_window

    def _get_learning_objective_classes(self):
        return [HierarchicalMaskedLanguageModelLearningObjective]

    def _calculate_step(self, num_of_visits):
        """
        Calculate the number of steps used for the sliding window strategy
        :param num_of_visits:
        :return:
        """
        if num_of_visits <= self._max_num_of_visits:
            return 1
        else:
            return math.ceil((num_of_visits - self._max_num_of_visits) /
                             self._sliding_window) + 1

    def _create_iterator(self):
        """
        Create an iterator that will iterate forever
        :return:
        """
        while True:
            for row in self._training_data.itertuples():
                # Skip the patient that doesn't have the min number of concepts
                if row.num_of_concepts >= self._min_num_of_concepts:
                    # Use a sliding window to slice out a portion of the medical history
                    for step in range(self._calculate_step(row.num_of_visits)):
                        end_index = row.num_of_visits - step * self._sliding_window
                        start_index = max(end_index - self._max_num_of_visits,
                                          0)
                        yield RowSlicer(row, start_index, end_index)

    def estimate_data_size(self):
        return self._training_data[
            self._training_data.num_of_concepts >= self._min_num_of_concepts].num_of_visits.apply(
            self._calculate_step).sum()


class HierarchicalBertMultiTaskDataGenerator(HierarchicalBertDataGenerator):
    def __init__(self, visit_tokenizer: ConceptTokenizer, *args, **kwargs):
        self._visit_tokenizer = visit_tokenizer
        super(HierarchicalBertMultiTaskDataGenerator,
              self).__init__(visit_tokenizer=self._visit_tokenizer,
                             *args,
                             **kwargs)

    def _get_learning_objective_classes(self):
        return [
            HierarchicalMaskedLanguageModelLearningObjective,
            HierarchicalBertSecondaryLearningObjective
        ]


class MedBertDataGenerator(BertDataGenerator):
    def _get_learning_objective_classes(self):
        return [
            MaskedLanguageModelLearningObjective,
            ProlongedLengthStayLearningObjective
        ]


class TemporalBertDataGenerator(BertDataGenerator):
    def __init__(self, time_window_size, *args, **kwargs):
        super(TemporalBertDataGenerator, self).__init__(*args, **kwargs)
        self._time_window_size = time_window_size

    def _create_iterator(self):
        """
        Create an iterator that will iterate forever
        :return:
        """
        while True:
            for row in self._training_data.itertuples():
                seq_length = len(row.token_ids)
                if self._is_training:
                    cursor = random.randint(0, seq_length -
                                            1) if self._is_random_cursor & (
                            seq_length > self._max_seq_len
                    ) else seq_length // 2

                    # Only include the concepts whose time stamps are within -half_time_window and
                    # half_time_window from the target time stamp
                    start_index, end_index = create_indexes_by_time_window(
                        row.dates, cursor, self._max_seq_len,
                        self._time_window_size)
                    if start_index < end_index:
                        yield RowSlicer(row, start_index, end_index)
                else:
                    yield RowSlicer(row, 0, seq_length)


class TemporalVisitPredictionBertDataGenerator(TemporalBertDataGenerator):
    def _get_learning_objective_classes(self):
        return [
            MaskedLanguageModelLearningObjective,
            VisitPredictionLearningObjective
        ]


class TimeAttentionDataGenerator(AbstractDataGeneratorBase):
    def __init__(self, concept_tokenizer: ConceptTokenizer,
                 time_window_size: int, *args, **kwargs):
        super(TimeAttentionDataGenerator,
              self).__init__(concept_tokenizer=concept_tokenizer,
                             time_window_size=time_window_size,
                             *args,
                             **kwargs)
        self._concept_tokenizer = concept_tokenizer
        self._time_window_size = time_window_size

    def _get_learning_objective_classes(self):
        return [TimeAttentionLearningObjective]

    def _create_iterator(self):
        """
        Create an iterator that will iterate forever
        :return:
        """
        while True:
            for row in self._training_data.itertuples():
                concept_ids, dates = zip(*sorted(zip(row.token_ids, row.dates),
                                                 key=lambda tup2: tup2[1]))
                for i in range(len(concept_ids)):
                    # Only include the concepts whose time stamps are within -half_time_window and
                    # half_time_window from the target time stamp
                    start_index, end_index = create_indexes_by_time_window(
                        dates, i, self._max_seq_len, self._time_window_size)
                    if start_index < end_index:
                        yield RowSlicer(row, start_index, end_index, i)

    def estimate_data_size(self):
        return len(self._training_data.token_ids.explode())


class FineTuningHierarchicalBertDataGenerator(HierarchicalBertDataGenerator):
    def _get_learning_objective_classes(self):
        return [HierarchicalMaskedLanguageModelLearningObjective,
                DemographicsLearningObjective,
                BertFineTuningLearningObjective]
