import datetime
from enum import Enum
from abc import abstractmethod, ABC
from typing import Dict, List, Any, Union
import collections
import copy

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas import Series
from datasets.formatting.formatting import LazyBatch

from meds.schema import birth_code, death_code
from spark_apps.decorators.patient_event_decorator import get_att_function
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from models.hf_models.tokenization_hf_cehrgpt import CehrGptTokenizer
from runner.hf_runner_argument_dataclass import DataTrainingArguments

birth_codes = [birth_code, "MEDS_BIRTH"]
death_codes = [death_code, "MEDS_DEATH"]

# OMOP concept ids for inpatient related visits
INPATIENT_VISIT_TYPES = [
    '9201', '262', '8971', '8920', '38004311'
]
INPATIENT_VISIT_TYPE_CODES = [
    'Visit/IP', 'Visit/ERIP', 'Visit/51', 'Visit/61', 'NUCC/315D00000X'
]
DISCHARGE_FACILITY_TYPES = [
    '8536', '8863', '44814650', '4161979', '38004519', '4216643', '8717', '8920', '4021968',
    '8546', '8971', '8970', '44814649', '8827', '8676', '38003619', '8870', '4146681'
]

DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class TruncationType(Enum):
    RANDOM_COMPLETE = "random_complete"
    RANDOM_RIGHT_TRUNCATION = "random_right_truncation"
    RANDOM_TRUNCATION = "random_truncation"
    TAIL = 'tail'


class DatasetMapping(ABC):

    def batch_transform(
            self,
            records: Union[LazyBatch, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if isinstance(records, LazyBatch):
            dataframe = records.pa_table.to_pandas()
        else:
            dataframe = pd.DataFrame(records)
        applied_dataframe = dataframe.apply(self.transform_pandas_series, axis=1)
        return applied_dataframe.to_dict(orient='list')

    def transform_pandas_series(self, series: Series) -> Series:
        record = self.transform(series.to_dict())
        return pd.Series(record)

    def remove_columns(self):
        return []

    @abstractmethod
    def transform(
            self,
            record: Dict[str, Any]
    ) -> Union[Dict[str, Any], Series]:
        """
        Transform the record
        Args
            record: The row to process, as generated by the CDM processing
        Returns
            A dictionary from names to numpy arrays to be used by pytorch.
        """
        pass


class MedToCehrBertDatasetMapping(DatasetMapping):
    def __init__(
            self,
            data_args: DataTrainingArguments,
            is_pretraining: bool = True
    ):
        self._time_token_function = get_att_function(data_args.att_function_type)
        self._include_auxiliary_token = data_args.include_auxiliary_token
        self._inpatient_time_token_function = get_att_function(data_args.inpatient_att_function_type)
        self._include_demographic_prompt = data_args.include_demographic_prompt
        self._is_pretraining = is_pretraining

    """
    This mapping function converts the MED (https://github.com/Medical-Event-Data-Standard/meds/tree/main) extension
    to the CehrBert format. We make several assumptions
    - The first event contains the demographic information
    - From the second event onward
        - the time of the event is visit_start_datetime.
        - the first measurement contains the code indicating a standard OMOP Visit concept_id (e.g. 9201, 9202)
        - in case of inpatient visits, the last measurement is assumed to
            contain the standard OMOP concept id for discharge facilities (e.g 8536)
        - in case of inpatient visits, datetime_value of the last measurement stores visit_end_datetime
    """

    def remove_columns(self):
        if self._is_pretraining:
            return ["visits", "patient_id", "birth_datetime", "index_date"]
        else:
            return ["visits", "patient_id", "birth_datetime", "index_date",
                    "visit_concept_ids", "num_of_concepts", "num_of_visits"]

    @staticmethod
    def _update_cehrbert_record(
            cehrbert_record: Dict[str, Any],
            code: str,
            visit_segment: int = 0,
            date: int = 0,
            age: int = -1,
            visit_concept_order: int = 0,
            visit_concept_id: str = '0',
            concept_value_mask: int = 0,
            concept_value: float = -1.,
            mlm_skip_value: int = 0,
    ) -> None:
        cehrbert_record['concept_ids'].append(code)
        cehrbert_record['visit_concept_orders'].append(visit_concept_order)
        cehrbert_record['ages'].append(age)
        cehrbert_record['dates'].append(date)
        cehrbert_record['visit_segments'].append(visit_segment)
        cehrbert_record['visit_concept_ids'].append(visit_concept_id)
        cehrbert_record['concept_value_masks'].append(concept_value_mask)
        cehrbert_record['concept_values'].append(concept_value)
        cehrbert_record['mlm_skip_values'].append(mlm_skip_value)

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
        # Extract the demographic information
        birth_datetime = record['birth_datetime']
        if isinstance(birth_datetime, pd.Timestamp):
            birth_datetime = birth_datetime.to_pydatetime()
        gender = record['gender']
        race = record['race']

        if self._include_demographic_prompt:
            first_visit = record['visits'][0]
            year_str = f'year:{str(first_visit["visit_start_datetime"].year)}'
            age_str = f'age:{str(relativedelta(first_visit["visit_start_datetime"], birth_datetime).years)}'

            self._update_cehrbert_record(cehrbert_record, year_str)
            self._update_cehrbert_record(cehrbert_record, age_str)
            self._update_cehrbert_record(cehrbert_record, gender)
            self._update_cehrbert_record(cehrbert_record, race)

        # A bool indicator to toggle between 1 and 2
        visit_segment_indicator = False

        # Use a data cursor to keep track of time
        date_cursor = None

        # Loop through all the visits excluding the first event containing the demographics
        for i, visit in enumerate(sorted(record['visits'], key=lambda e: e['visit_start_datetime'])):

            events = visit['events']

            # Skip this visit if the number measurements in the event is zero
            if events is None or len(events) == 0:
                continue

            visit_start_datetime = visit['visit_start_datetime']
            time_delta = (visit_start_datetime - date_cursor).days if date_cursor else None
            date_cursor = visit_start_datetime

            # We assume the first measurement to be the visit type of the current visit
            visit_type = visit['visit_type']
            is_inpatient = visit_type in INPATIENT_VISIT_TYPES or visit_type in INPATIENT_VISIT_TYPE_CODES

            # Add artificial time tokens to the patient timeline if timedelta exists
            if time_delta:
                # This generates an artificial time token depending on the choice of the time token functions
                self._update_cehrbert_record(
                    cehrbert_record,
                    code=self._time_token_function(time_delta),
                    visit_concept_order=i + 1
                )

            # Add the VS token to the patient timeline to mark the start of a visit
            age = relativedelta(visit['visit_start_datetime'], birth_datetime).years
            # Calculate the week number since the epoch time
            date = (visit['visit_start_datetime'] - datetime.datetime(year=1970, month=1, day=1)).days // 7
            visit_segment = int(visit_segment_indicator) + 1

            self._update_cehrbert_record(
                cehrbert_record,
                code='[VS]',
                visit_concept_order=i + 1,
                age=age,
                date=date,
                visit_segment=visit_segment,
                visit_concept_id=visit_type
            )

            if self._include_auxiliary_token:
                self._update_cehrbert_record(
                    cehrbert_record,
                    code=visit_type,
                    visit_concept_order=i + 1,
                    age=age,
                    date=date,
                    visit_segment=visit_segment,
                    visit_concept_id=visit_type
                )

            for e in events:
                # If the event doesn't have a time stamp, we skip it
                if not e['time']:
                    continue
                # Add a medical token to the patient timeline
                # If this is an inpatient visit, we use the event time stamps to calculate age and date
                # because the patient can stay in the hospital for a period of time.
                if is_inpatient:
                    # Calculate age using the event time stamp
                    age = relativedelta(e['time'], birth_datetime).years
                    # Calculate the week number since the epoch time
                    date = (e['time'] - datetime.datetime(year=1970, month=1, day=1)).days // 7
                else:
                    # For outpatient visits, we use the visit time stamp to calculate age and time because we assume
                    # the outpatient visits start and end on the same day
                    pass

                # Calculate the time diff in days w.r.t the previous measurement
                meas_time_diff = relativedelta(e['time'], date_cursor).days
                # Update the date_cursor if the time diff between two neighboring measurements is greater than and
                # equal to 1 day
                if meas_time_diff > 0:
                    date_cursor = e['time']
                    if self._inpatient_time_token_function:
                        # This generates an artificial time token depending on the choice of the time token functions
                        self._update_cehrbert_record(
                            cehrbert_record,
                            code=f'i-{self._inpatient_time_token_function(meas_time_diff)}',
                            visit_concept_order=i + 1,
                            visit_segment=visit_segment,
                            visit_concept_id=visit_type
                        )

                # If numeric_value exists, this is a concept/value tuple, we indicate this using a concept_value_mask
                concept_value_mask = int(e['numeric_value'] is not None)
                concept_value = e['numeric_value'] if concept_value_mask == 1 else -1

                self._update_cehrbert_record(
                    cehrbert_record,
                    code=e['code'],
                    age=age,
                    date=date,
                    visit_concept_order=i + 1,
                    visit_segment=visit_segment,
                    visit_concept_id=visit_type,
                    concept_value_mask=concept_value_mask,
                    concept_value=concept_value,
                    mlm_skip_value=concept_value_mask
                )

            if is_inpatient:
                # If visit_end_datetime is populated for the inpatient visit, we update the date_cursor
                visit_end_datetime = visit.get('visit_end_datetime', None)
                if visit_end_datetime:
                    date_cursor = visit_end_datetime

                if self._include_auxiliary_token:
                    # Reuse the age and date calculated for the last event in the patient timeline for the discharge
                    # facility event
                    discharge_facility = (
                        visit['discharge_facility'] if ('discharge_facility' in visit) and visit['discharge_facility']
                        else '0'
                    )

                    self._update_cehrbert_record(
                        cehrbert_record,
                        code=discharge_facility,
                        age=age,
                        date=date,
                        visit_concept_order=i + 1,
                        visit_segment=visit_segment,
                        visit_concept_id=visit_type
                    )

            # Reuse the age and date calculated for the last event in the patient timeline
            self._update_cehrbert_record(
                cehrbert_record,
                code='[VE]',
                age=age,
                date=date,
                visit_concept_order=i + 1,
                visit_segment=visit_segment,
                visit_concept_id=visit_type
            )

            # Toggle visit_segment_indicator
            visit_segment_indicator = not visit_segment_indicator

        # Generate the orders of the concepts that the cehrbert dataset mapping function expects
        cehrbert_record['orders'] = list(range(1, len(cehrbert_record['concept_ids']) + 1))

        # Add some count information for this sequence
        cehrbert_record['num_of_concepts'] = len(cehrbert_record['concept_ids'])
        cehrbert_record['num_of_visits'] = len(record['visits'])

        if 'label' in record:
            cehrbert_record['label'] = record['label']
        if 'age_at_index' in record:
            cehrbert_record['age_at_index'] = record['age_at_index']

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
        if sorting_columns is None:
            sorting_columns = record.get('dates', None)

        if sorting_columns is None:
            return record

        sorting_columns = list(map(int, sorting_columns))
        seq_length = len(record['concept_ids'])
        column_names = ['concept_ids']
        column_values = [record['concept_ids']]

        for k, v in record.items():
            if k in column_names:
                continue
            if isinstance(v, (list, np.ndarray)) and len(v) == seq_length:
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


class HFTokenizationMapping(DatasetMapping):
    def __init__(
            self,
            concept_tokenizer: CehrBertTokenizer,
            is_pretraining: bool
    ):
        self._concept_tokenizer = concept_tokenizer
        self._is_pretraining = is_pretraining
        self._lab_token_ids = self._concept_tokenizer.lab_token_ids

    def remove_columns(self):
        if self._is_pretraining:
            return ["concept_ids", "orders"]
        else:
            return ["concept_ids", "mlm_skip_values", "orders"]

    def transform(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:

        input_ids = self._concept_tokenizer.encode(record['concept_ids'])
        record['input_ids'] = input_ids
        concept_value_masks = record['concept_value_masks']
        concept_values = record['concept_values']

        # If any concept has a value associated with it, we normalize the value
        if np.any(np.asarray(concept_value_masks) > 0):
            normalized_concept_values = copy.deepcopy(concept_values)
            for i, (concept_id, token_id, concept_value_mask, concept_value) in enumerate(
                    zip(record['concept_ids'], input_ids, concept_value_masks, concept_values)
            ):
                if token_id in self._lab_token_ids:
                    normalized_concept_value = self._concept_tokenizer.normalize(concept_id, concept_value)
                    normalized_concept_values[i] = normalized_concept_value
            record['concept_values'] = normalized_concept_values

        # If mlm_skip_value=1, this indicates there is a value associated with this position and
        # hence we block the MLM to randomly pick this token to be predicted
        if self._is_pretraining:
            if 'mlm_skip_values' in record:
                labels = copy.deepcopy(input_ids)
                mlm_skip_values = record['mlm_skip_values']

                assert len(input_ids) == len(mlm_skip_values), \
                    f"The following equality must be true: len(input_ids) == len(mlm_skip_values)"

                for i, (input_id, mlm_skip_value) in enumerate(zip(input_ids, mlm_skip_values)):
                    if mlm_skip_value == 1:
                        labels[i] = -100

                record.update({
                    'input_ids': input_ids,
                    'labels': labels
                })

        return record


class HFFineTuningMapping(DatasetMapping):
    def transform(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:
        new_record = copy.deepcopy(record)
        new_record.update({
            'age_at_index': record['age_at_index'],
            'classifier_label': record['label']
        })
        return new_record


class HFCehrGptTokenizationMapping(DatasetMapping):
    def __init__(
            self,
            concept_tokenizer: CehrGptTokenizer,
    ):
        self._concept_tokenizer = concept_tokenizer
        self._lab_token_ids = self._concept_tokenizer.lab_token_ids

    def transform(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:
        input_ids = self._concept_tokenizer.encode(record['concept_ids'])
        record['input_ids'] = input_ids
        concept_value_masks = record['concept_value_masks']
        concept_values = record['concept_values']

        # If any concept has a value associated with it, we normalize the value
        if np.any(concept_value_masks > 0):
            normalized_concept_values = copy.deepcopy(concept_values)
            for i, (concept_id, token_id, concept_value_mask, concept_value) in enumerate(
                    zip(record['concept_ids'], input_ids, concept_value_masks, concept_values)
            ):
                if token_id in self._lab_token_ids:
                    normalized_concept_value = self._concept_tokenizer.normalize(concept_id, concept_value)
                    normalized_concept_values[i] = normalized_concept_value
            record['concept_values'] = normalized_concept_values
        return record
