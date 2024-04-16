import logging
import datetime
from enum import Enum
from abc import abstractmethod, ABC
from typing import Dict, Any
import collections
import random
import numpy as np
import copy

from med_extension.schema_extension import get_measurements_from_visit

from dateutil.relativedelta import relativedelta
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer

# OMOP concept ids for inpatient related visits
INPATIENT_VISIT_TYPES = [
    '9201', '262', '8971', '8920', '38004311'
]
DISCHARGE_FACILITY_TYPES = [
    '8536', '8863', '44814650', '4161979', '38004519', '4216643', '8717', '8920', '4021968',
    '8546', '8971', '8970', '44814649', '8827', '8676', '38003619', '8870', '4146681'
]


class TruncationType(Enum):
    RANDOM = 'random'
    TAIL = 'tail'


class DatasetMapping(ABC):

    @abstractmethod
    def transform(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform the record
        Args
            record: The row to process, as generated by the CDM processing
        Returns
            A dictionary from names to numpy arrays to be used by pytorch.
        """
        pass


class MedToCehrBertDatasetMapping(DatasetMapping):
    """
    This mapping function converts the MED (https://github.com/Medical-Event-Data-Standard/meds/tree/main)
    to the CehrBert format. We make several assumptions
    - The first event contains the demographic information
    - From the second event onward
        - the time of the event is visit_start_datetime.
        - the first measurement contains the code indicating a standard OMOP Visit concept_id (e.g. 9201, 9202)
        - in case of inpatient visits, the last measurement is assumed to
            contain the standard OMOP concept id for discharge facilities (e.g 8536)
        - in case of inpatient visits, datetime_value of the last measurement stores visit_end_datetime
    """

    def __init__(
            self,
            time_token_function,
            include_inpatient_att: bool = False,
            inpatient_time_token_function=None,
            include_demographic_prompt: bool = False
    ):
        self._time_token_function = time_token_function
        self._include_inpatient_att = include_inpatient_att
        if include_inpatient_att and not inpatient_time_token_function:
            raise ValueError('inpatient_time_token_function needs to be provided when include_inpatient_att is True')
        self._inpatient_time_token_function = inpatient_time_token_function
        self._include_demographic_prompt = include_demographic_prompt

    def transform(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:

        cehrbert_record = {
            'person_id': record['patient_id'],
            'concept_ids': [],
            'visit_segments': [],
            'orders': [],
            'dates': [],
            'ages': [],
            'visit_concept_orders': [],
            'concept_value_masks': [],
            'concept_values': [],
            'mlm_skip_values': [],
            'visit_concept_ids': []
        }
        # At least one visit should exist
        assert len(record['visits']) >= 1

        # Extract the demographic information
        birth_datetime = record['birth_datetime']
        gender = record['gender']
        race = record['race']

        if self._include_demographic_prompt:
            first_visit = record['visits'][0]
            year_str = f'year:{str(first_visit["visit_start_datetime"].year)}'
            age_str = f'age:{str(relativedelta(first_visit["visit_start_datetime"], birth_datetime).years)}'
            cehrbert_record['concept_ids'].extend([year_str, age_str, gender, race])
            cehrbert_record['ages'].extend([-1, -1, -1, -1])
            cehrbert_record['dates'].extend([0, 0, 0, 0])
            cehrbert_record['visit_concept_orders'].extend([0, 0, 0, 0])
            cehrbert_record['visit_segments'].extend([0, 0, 0, 0])
            cehrbert_record['visit_concept_ids'].extend(['0', '0', '0', '0'])
            cehrbert_record['concept_value_masks'].extend([0, 0, 0, 0])
            cehrbert_record['concept_values'].extend([-1, -1, -1, -1])
            cehrbert_record['mlm_skip_values'].extend([0, 0, 0, 0])

        # A bool indicator to toggle between 1 and 2
        visit_segment_indicator = False

        # Use a data cursor to keep track of time
        date_cursor = None

        # Loop through all the visits excluding the first event containing the demographics
        for i, visit in enumerate(sorted(record['visits'], key=lambda e: e['visit_start_datetime'])):

            measurements = get_measurements_from_visit(visit)

            # Skip this visit if the number measurements in the event is zero
            if not measurements:
                continue

            visit_start_datetime = visit['visit_start_datetime']
            time_delta = (visit_start_datetime - date_cursor).days if date_cursor else None
            date_cursor = visit_start_datetime

            # We assume the first measurement to be the visit type of the current visit
            visit_type = visit['visit_type']
            is_inpatient = visit_type in INPATIENT_VISIT_TYPES

            # Add artificial time tokens to the patient timeline if timedelta exists
            if time_delta:
                # This generates an artificial time token depending on the choice of the time token functions
                att_token = self._time_token_function(time_delta)
                cehrbert_record['concept_ids'].append(att_token)
                cehrbert_record['ages'].append(-1)
                cehrbert_record['dates'].append(0)
                cehrbert_record['visit_concept_orders'].append(i + 1)
                cehrbert_record['visit_segments'].append(0)
                cehrbert_record['visit_concept_ids'].append('0')
                cehrbert_record['concept_value_masks'].append(0)
                cehrbert_record['concept_values'].append(-1)
                cehrbert_record['mlm_skip_values'].append(0)

            # Add the VS token to the patient timeline to mark the start of a visit
            age = relativedelta(visit['visit_start_datetime'], birth_datetime).years
            # Calculate the week number since the epoch time
            date = (visit['visit_start_datetime'] - datetime.datetime(year=1970, month=1, day=1)).days // 7
            visit_segment = int(visit_segment_indicator) + 1

            cehrbert_record['concept_ids'].extend(['[VS]', visit_type])
            cehrbert_record['ages'].extend([age] * 2)
            cehrbert_record['dates'].extend([date] * 2)
            cehrbert_record['visit_concept_orders'].extend([i + 1] * 2)
            cehrbert_record['visit_segments'].extend([visit_segment] * 2)
            cehrbert_record['visit_concept_ids'].extend([visit_type] * 2)
            cehrbert_record['concept_value_masks'].extend([0] * 2)
            cehrbert_record['concept_values'].extend([-1] * 2)
            cehrbert_record['mlm_skip_values'].extend([0] * 2)

            # Sort all measurements using time, in case of a tie, we use the natural order of codes to tiebreak
            for m_i, m in enumerate(sorted(measurements, key=lambda m: (m['datetime_value'], m['code']))):
                # Add a medical token to the patient timeline
                # If this is an inpatient visit, we use the event time stamps to calculate age and date
                # because the patient can stay in the hospital for a period of time.
                if is_inpatient:
                    # Calculate age using the event time stamp
                    age = relativedelta(m['datetime_value'], birth_datetime).years
                    # Calculate the week number since the epoch time
                    date = (m['datetime_value'] - datetime.datetime(year=1970, month=1, day=1)).days // 7
                else:
                    # For outpatient visits, we use the visit time stamp to calculate age and time because we assume
                    # the outpatient visits start and end on the same day
                    pass

                # Calculate the time diff in days w.r.t the previous measurement
                meas_time_diff = relativedelta(m['datetime_value'], date_cursor).days
                # Update the date_cursor if the time diff between two neighboring measurements is greater than and
                # equal to 1 day
                if meas_time_diff > 0:
                    date_cursor = m['datetime_value']
                    if self._include_inpatient_att:
                        # This generates an artificial time token depending on the choice of the time token functions
                        att_token = f'i-{self._inpatient_time_token_function(time_delta)}'
                        cehrbert_record['concept_ids'].append(att_token)
                        cehrbert_record['ages'].append(-1)
                        cehrbert_record['dates'].append(0)
                        cehrbert_record['visit_concept_orders'].append(i + 1)
                        cehrbert_record['visit_segments'].append(visit_segment)
                        cehrbert_record['visit_concept_ids'].append(visit_type)
                        cehrbert_record['concept_value_masks'].append(0)
                        cehrbert_record['concept_values'].append(-1)
                        cehrbert_record['mlm_skip_values'].append(0)

                # If numeric_value exists, this is a concept/value tuple, we indicate this using a concept_value_mask
                concept_value_mask = int('numeric_value' in m)
                concept_value = m['numeric_value'] if 'numeric_value' in m else -1

                cehrbert_record['concept_ids'].append(m['code'])
                cehrbert_record['ages'].append(age)
                cehrbert_record['dates'].append(date)
                cehrbert_record['visit_concept_orders'].append(i + 1)
                cehrbert_record['visit_segments'].append(visit_segment)
                cehrbert_record['visit_concept_ids'].append(visit_type)
                cehrbert_record['concept_value_masks'].append(concept_value_mask)
                cehrbert_record['concept_values'].append(concept_value)
                cehrbert_record['mlm_skip_values'].append(int('numeric_value' in m))

            if is_inpatient:
                # If visit_end_datetime is populated for the inpatient visit, we update the date_cursor
                if 'visit_end_datetime' in visit:
                    date_cursor = visit['visit_end_datetime']
                # Reuse the age and date calculated for the last event in the patient timeline
                discharge_facility = visit['discharge_facility'] if 'discharge_facility' in visit else '0'
                cehrbert_record['concept_ids'].append(discharge_facility)
                cehrbert_record['ages'].append(age)
                cehrbert_record['dates'].append(date)
                cehrbert_record['visit_concept_orders'].append(i + 1)
                cehrbert_record['visit_segments'].append(visit_segment)
                cehrbert_record['visit_concept_ids'].append(visit_type)
                cehrbert_record['concept_value_masks'].append(0)
                cehrbert_record['concept_values'].append(-1)
                cehrbert_record['mlm_skip_values'].append(0)

            # Reuse the age and date calculated for the last event in the patient timeline
            cehrbert_record['concept_ids'].append('[VE]')
            cehrbert_record['ages'].append(age)
            cehrbert_record['dates'].append(date)
            cehrbert_record['visit_concept_orders'].append(i + 1)
            cehrbert_record['visit_segments'].append(visit_segment)
            cehrbert_record['visit_concept_ids'].append(visit_type)
            cehrbert_record['concept_value_masks'].append(0)
            cehrbert_record['concept_values'].append(-1)
            cehrbert_record['mlm_skip_values'].append(0)

            # Toggle visit_segment_indicator
            visit_segment_indicator = not visit_segment_indicator

        # Generate the orders of the concepts that the cehrbert dataset mapping function expects
        cehrbert_record['orders'] = list(range(1, len(cehrbert_record['concept_ids']) + 1))

        # Add some count information for this sequence
        cehrbert_record['num_of_concepts'] = len(cehrbert_record['concept_ids'])
        cehrbert_record['num_of_visits'] = len(record['visits'])

        # Add demographics for this patient
        cehrbert_record['birth_datetime'] = birth_datetime
        cehrbert_record['gender'] = gender
        cehrbert_record['race'] = race

        return cehrbert_record


class SortPatientSequenceMapping(DatasetMapping):
    """
    A mapping function to order all the features using a pre-defined orders/dates column.
    This may not be necessary since the order is feature columns should've been ordered
    correctly during the data generation process in the spark application. However,
    it's a good idea to sort them explicitly one more time
    """

    def transform(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sort all the list features using a pre-defined orders/dates. If orders/dates columns are not provided,
        do nothing.
        """

        sorting_columns = record.get('orders', None)
        if not sorting_columns:
            sorting_columns = record.get('dates', None)

        if not sorting_columns:
            return record

        sorting_columns = list(map(int, sorting_columns))
        seq_length = len(record['concept_ids'])
        column_names = ['concept_ids']
        column_values = [record['concept_ids']]

        for k, v in record.items():
            if k in column_names:
                continue
            if isinstance(v, list) and len(v) == seq_length:
                column_names.append(k)
                column_values.append(v)

        sorted_list = sorted(zip(sorting_columns, *column_values), key=lambda tup2: (tup2[0], tup2[1]))

        # uses a combination of zip() and unpacking (*) to transpose the list of tuples. This means converting rows
        # into columns: the first tuple formed from all the first elements of the sorted tuples, the second tuple
        # from all the second elements, and so on. Then slices the resulting list of tuples to skip the first tuple
        # (which contains the sorting criteria) and retain only the data columns.
        sorted_features = list(zip(*list(sorted_list)))[1:]
        new_record = collections.OrderedDict()
        for i, new_val in enumerate(sorted_features):
            new_record[column_names[i]] = list(new_val)
        return new_record


class GenerateStartEndIndexMapping(DatasetMapping):
    def __init__(
            self,
            max_sequence_length,
            truncate_type=TruncationType.RANDOM
    ):
        self._max_sequence_length = max_sequence_length
        self._truncate_type = truncate_type

    def transform(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapted from https://github.com/OHDSI/Apollo/blob/main/data_loading/data_transformer.py

        Adding the start and end indices to extract a portion of the patient sequence
        """

        seq_length = len(record['concept_ids'])
        new_max_length = self._max_sequence_length - 1  # Subtract one for the [CLS] token
        if seq_length > new_max_length and self._truncate_type == TruncationType.RANDOM:
            start_index = random.randint(0, seq_length - new_max_length)
            end_index = min(seq_length, start_index + new_max_length)
            record['start_index'] = start_index
            record['end_index'] = end_index
        else:
            record['start_index'] = max(0, seq_length - new_max_length)
            record['end_index'] = seq_length
        return record


class HFMaskedLanguageModellingMapping(DatasetMapping):
    def __init__(
            self,
            concept_tokenizer: CehrBertTokenizer,
            is_pretraining: bool
    ):
        self._concept_tokenizer = concept_tokenizer
        self._is_pretraining = is_pretraining

    def transform(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:

        if 'start_index' not in record:
            raise ValueError('Missing start_index in row')

        if 'end_index' not in record:
            raise ValueError('Missing end_index in row')

        start_index = record['start_index']
        end_index = record['end_index']

        seq_length = len(record['concept_ids'])
        new_record = collections.OrderedDict()
        for k, v in record.items():
            if isinstance(v, list) and len(v) == seq_length:
                new_record[k] = v[start_index:end_index]

        input_ids = self._concept_tokenizer.encode(new_record['concept_ids'])

        new_record.update({
            'input_ids': input_ids
        })

        if self._is_pretraining:
            masked_input_ids, output_mask = self._mask_concepts(input_ids, new_record['mlm_skip_values'])
            masks = np.empty_like(masked_input_ids, dtype=np.int32)
            # -100 is ignored by the torch CrossEntropyLoss
            masks.fill(-100)
            labels = np.where(output_mask == 1, input_ids, masks)
            new_record.update({
                'input_ids': masked_input_ids.tolist(),
                'labels': labels.tolist()
            })

        return new_record

    def _mask_concepts(self, concepts, mlm_skip_values):
        """
        Mask out 15% of the concepts

        :param concepts:
        :param mlm_skip_values:
        :return:
        """

        masked_concepts = np.asarray(concepts).copy()
        output_mask = np.zeros((len(concepts),), dtype=int)

        for word_pos in range(0, len(concepts)):
            # Check if this position needs to be skipped
            if mlm_skip_values[word_pos] == 1:
                continue
            if concepts[word_pos] == self._concept_tokenizer.unused_token_index:
                break
            if random.random() < 0.15:
                dice = random.random()
                if dice < 0.8:
                    masked_concepts[word_pos] = self._concept_tokenizer.mask_token_index
                elif dice < 0.9:
                    masked_concepts[word_pos] = random.randint(
                        0,
                        self._concept_tokenizer.vocab_size - 1
                    )
                # else: 10% of the time we just leave the word as is
                output_mask[word_pos] = 1

        return masked_concepts, output_mask


class HFFineTuningMapping(DatasetMapping):
    def transform(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:
        if 'start_index' not in record:
            raise ValueError('Missing start_index in row')

        if 'end_index' not in record:
            raise ValueError('Missing end_index in row')

        new_record = copy.deepcopy(record)
        new_record.update({
            'age_at_index': record['age'],
            'classifier_label': record['label']
        })
        return new_record
