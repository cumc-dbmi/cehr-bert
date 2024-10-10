import collections
import functools
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple, Union

import meds_reader
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Split

from cehrbert.data_generators.hf_data_generator import (
    DEFAULT_ED_CONCEPT_ID,
    DEFAULT_INPATIENT_CONCEPT_ID,
    DEFAULT_OUTPATIENT_CONCEPT_ID,
    UNKNOWN_VALUE,
)
from cehrbert.data_generators.hf_data_generator.hf_dataset import apply_cehrbert_dataset_mapping
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import MedToCehrBertDatasetMapping
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    MedsToCehrBertConversion,
)
from cehrbert.med_extension.schema_extension import CehrBertPatient, Event, Visit
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments, MedsToCehrBertConversionType

MEDS_SPLIT_DATA_SPLIT_MAPPING = {
    "train": Split.TRAIN,
    "tuning": Split.VALIDATION,
    "held_out": Split.TEST,
}
NON_ALPHANUMERIC_CHARS = r"[\w\/\\:\-_]"


def get_meds_to_cehrbert_conversion_cls(
    meds_to_cehrbert_conversion_type: Union[MedsToCehrBertConversionType, str], **kwargs
) -> MedsToCehrBertConversion:
    for cls in MedsToCehrBertConversion.__subclasses__():
        if isinstance(meds_to_cehrbert_conversion_type, MedsToCehrBertConversionType):
            if meds_to_cehrbert_conversion_type.name == cls.__name__:
                return cls(**kwargs)
        elif isinstance(meds_to_cehrbert_conversion_type, str):
            if meds_to_cehrbert_conversion_type == cls.__name__:
                return cls(**kwargs)
    raise RuntimeError(f"{meds_to_cehrbert_conversion_type} is not a valid MedsToCehrBertConversionType")


def get_subject_split(meds_reader_db_path: str) -> Dict[str, List[int]]:
    patient_split = pd.read_parquet(os.path.join(meds_reader_db_path, "metadata/subject_splits.parquet"))
    result = {str(group): records["subject_id"].tolist() for group, records in patient_split.groupby("split")}
    return result


@dataclass
class PatientDemographics:
    birth_datetime: datetime = None
    race: str = None
    gender: str = None
    ethnicity: str = None


class PatientBlock:
    """
    Represents a block of medical events for a single patient visit, including.

    inferred visit type and various admission and discharge statuses.

    Attributes:
        visit_id (int): The unique ID of the visit.
        events (List[meds_reader.Event]): A list of medical events associated with this visit.
        min_time (datetime): The earliest event time in the visit.
        max_time (datetime): The latest event time in the visit.
        conversion (MedsToCehrBertConversion): Conversion object for mapping event codes to CEHR-BERT.
        has_ed_admission (bool): Whether the visit includes an emergency department (ED) admission event.
        has_admission (bool): Whether the visit includes an admission event.
        has_discharge (bool): Whether the visit includes a discharge event.
        visit_type (str): The inferred type of visit, such as inpatient, ED, or outpatient.
    """

    def __init__(
        self,
        events: List[meds_reader.Event],
        visit_id: int,
        conversion: MedsToCehrBertConversion,
    ):
        """
        Initializes a PatientBlock instance, inferring the visit type based on the events and caching.

        admission and discharge status.

        Args:
            events (List[meds_reader.Event]): The medical events associated with the visit.
            visit_id (int): The unique ID of the visit.
            conversion (MedsToCehrBertConversion): Conversion object for mapping event codes to CEHR-BERT.

        Attributes are initialized to store visit metadata and calculate admission/discharge statuses.
        """
        self.visit_id = visit_id
        self.events = events
        self.min_time = events[0].time
        self.max_time = events[-1].time
        self.conversion = conversion

        # Cache these variables so we don't need to compute
        self.has_ed_admission = self._has_ed_admission()
        self.has_admission = self._has_admission()
        self.has_discharge = self._has_discharge()

        # Infer the visit_type from the events
        # Admission takes precedence over ED
        if self.has_admission:
            self.visit_type = DEFAULT_INPATIENT_CONCEPT_ID
        elif self.has_ed_admission:
            self.visit_type = DEFAULT_ED_CONCEPT_ID
        else:
            self.visit_type = DEFAULT_OUTPATIENT_CONCEPT_ID

    def _has_ed_admission(self) -> bool:
        """
        Determines if the visit includes an emergency department (ED) admission event.

        Returns:
            bool: True if an ED admission event is found, False otherwise.
        """
        for event in self.events:
            for matching_rule in self.conversion.get_ed_admission_matching_rules():
                if re.match(matching_rule, event.code):
                    return True
        return False

    def _has_admission(self) -> bool:
        """
        Determines if the visit includes a hospital admission event.

        Returns:
            bool: True if an admission event is found, False otherwise.
        """
        for event in self.events:
            for matching_rule in self.conversion.get_admission_matching_rules():
                if re.match(matching_rule, event.code):
                    return True
        return False

    def _has_discharge(self) -> bool:
        """
        Determines if the visit includes a discharge event.

        Returns:
            bool: True if a discharge event is found, False otherwise.
        """
        for event in self.events:
            for matching_rule in self.conversion.get_discharge_matching_rules():
                if re.match(matching_rule, event.code):
                    return True
        return False

    def get_discharge_facility(self) -> Optional[str]:
        """
        Extracts the discharge facility code from the discharge event, if present.

        Returns:
            Optional[str]: The sanitized discharge facility code, or None if no discharge event is found.
        """
        if self._has_discharge():
            for event in self.events:
                for matching_rule in self.conversion.get_discharge_matching_rules():
                    if matching_rule in event.code:
                        discharge_facility = event.code.replace(matching_rule, "")
                        discharge_facility = re.sub(r"[^a-zA-Z]", "_", discharge_facility)
                        return discharge_facility
        return None

    def _convert_event(self, event) -> List[Event]:
        """
        Converts a medical event into a list of CEHR-BERT-compatible events, potentially parsing.

        numeric values from text-based events.

        Args:
            event (meds_reader.Event): The medical event to be converted.

        Returns:
            List[Event]: A list of converted events, possibly numeric, based on the original event's code and value.
        """
        code = event.code
        time = getattr(event, "time", None)
        text_value = getattr(event, "text_value", None)
        numeric_value = getattr(event, "numeric_value", None)
        unit = getattr(event, "unit", None)

        if numeric_value is None and text_value is not None:
            conversion_rule = self.conversion.get_text_event_to_numeric_events_rule(code)
            if conversion_rule:
                match = re.search(conversion_rule.parsing_pattern, text_value)
                if match:
                    if len(match.groups()) == len(conversion_rule.mapped_event_labels):
                        events = [
                            Event(
                                code=label,
                                time=time,
                                numeric_value=float(value),
                                unit=unit,
                                properties={"visit_id": self.visit_id, "table": "meds"},
                            )
                            for label, value in zip(conversion_rule.mapped_event_labels, match.groups())
                            if value.isnumeric()
                        ]
                        return events

        return [
            Event(
                code=code,
                time=time,
                numeric_value=numeric_value,
                unit=unit,
                text_value=text_value,
                properties={"visit_id": self.visit_id, "table": "meds"},
            )
        ]

    def get_meds_events(self) -> Iterable[Event]:
        """
        Retrieves all medication events for the visit, converting each raw event if necessary.

        Returns:
            Iterable[Event]: A list of CEHR-BERT-compatible medication events for the visit.
        """
        events = []
        for e in self.events:
            events.extend(self._convert_event(e))
        return events


def convert_one_patient(
    patient: meds_reader.Subject,
    conversion: MedsToCehrBertConversion,
    prediction_time: datetime = None,
    label: Union[int, float] = None,
) -> CehrBertPatient:
    """
    Converts a patient's event data into a CehrBertPatient object, processing.

    their medical history, visit details, and demographic information.

    Parameters:
    ----------
    patient : meds_reader.Subject
        The patient's event data, including time-stamped medical events such as
        demographic data (race, gender, ethnicity) and clinical visits (ED admissions,
        hospital admissions, discharges).

    conversion : MedsToCehrBertConversion
        The conversion object to map and process medical event data into the format
        required by CehrBert.

    default_visit_id : int, optional (default=1)
        The starting ID for patient visits. This is incremented as new visits are
        identified in the event data.

    prediction_time : datetime, optional (default=None)
        The cutoff time for processing events. Events occurring after this time are
        ignored.

    label : Union[int, float], optional (default=None)
        The prediction label associated with this patient, which could represent a
        clinical outcome (e.g., survival or treatment response).

    Returns:
    -------
    CehrBertPatient
        An object containing the patient's transformed event data, visits, demographics,
        and associated label in a structure compatible with CehrBert's input requirements.

    Description:
    -----------
    This function processes a patient's medical history, including demographic
    information (birth date, race, gender, and ethnicity) and visit details. It iterates
    through the patient's events and groups them into visits (ED, admission, discharge).
    Visits are formed based on timestamps, and certain logic is applied to merge ED visits
    into hospital admissions if they occur within 24 hours of each other.

    For each event, demographic attributes like birth date, race, gender, and ethnicity
    are extracted. If the event has a timestamp, it is compared with `prediction_time` to
    filter out events that occurred after the specified time.

    The function handles ongoing (incomplete) visits and cases where multiple visits
    should be merged (e.g., ED followed by hospital admission within 24 hours). After
    processing the events, visits are built with details such as visit type, start/end
    datetime, and events during the visit.

    The function returns a `CehrBertPatient` object that includes the patient's medical
    events, structured into visits, along with demographic information, and optionally
    a prediction label.

    Example Usage:
    -------------
    patient_data = convert_one_patient(
        patient=some_patient_object,
        conversion=some_conversion_object,
        default_visit_id=1,
        prediction_time=datetime.now(),
        label=1
    )
    """
    demographics, patient_blocks = conversion.generate_demographics_and_patient_blocks(
        patient=patient, prediction_time=prediction_time
    )

    patient_block_dict = collections.defaultdict(list)
    for patient_block in patient_blocks:
        patient_block_dict[patient_block.visit_id].append(patient_block)

    visits = list()
    for visit_id, blocks in patient_block_dict.items():
        visit_type = blocks[0].visit_type
        visit_start_datetime = min([b.min_time for b in blocks])
        visit_end_datetime = max([b.max_time for b in blocks])
        discharge_facility = (
            next(filter(None, [b.get_discharge_facility() for b in blocks]), None)
            if visit_type == DEFAULT_INPATIENT_CONCEPT_ID
            else None
        )
        visit_events = list()
        for block in blocks:
            visit_events.extend(block.get_meds_events())

        visits.append(
            Visit(
                visit_type=visit_type,
                visit_start_datetime=visit_start_datetime,
                visit_end_datetime=visit_end_datetime,
                discharge_facility=(discharge_facility if discharge_facility else UNKNOWN_VALUE),
                events=visit_events,
            )
        )
    age_at_index = -1
    if prediction_time is not None and demographics.birth_datetime is not None:
        age_at_index = prediction_time.year - demographics.birth_datetime.year
        if (prediction_time.month, prediction_time.day) < (
            demographics.birth_datetime.month,
            demographics.birth_datetime.day,
        ):
            age_at_index -= 1

    # birth_datetime can not be None
    assert (
        demographics.birth_datetime is not None
    ), f"patient_id: {patient.subject_id} does not have a valid birth_datetime"

    return CehrBertPatient(
        patient_id=patient.subject_id,
        birth_datetime=demographics.birth_datetime,
        visits=visits,
        race=demographics.race if demographics.race else UNKNOWN_VALUE,
        gender=demographics.gender if demographics.gender else UNKNOWN_VALUE,
        ethnicity=demographics.ethnicity if demographics.ethnicity else UNKNOWN_VALUE,
        index_date=prediction_time,
        age_at_index=age_at_index,
        label=label,
    )


def create_dataset_from_meds_reader(
    data_args: DataTrainingArguments,
    default_visit_id: int = 1,
    is_pretraining: bool = True,
) -> DatasetDict:
    train_dataset = _create_cehrbert_data_from_meds(
        data_args=data_args,
        split="train",
        default_visit_id=default_visit_id,
        is_pretraining=is_pretraining,
    )
    tuning_dataset = _create_cehrbert_data_from_meds(
        data_args=data_args,
        split="tuning",
        default_visit_id=default_visit_id,
        is_pretraining=is_pretraining,
    )
    held_out_dataset = _create_cehrbert_data_from_meds(
        data_args=data_args,
        split="held_out",
        default_visit_id=default_visit_id,
        is_pretraining=is_pretraining,
    )

    return DatasetDict({"train": train_dataset, "validation": tuning_dataset, "test": held_out_dataset})


def _meds_to_cehrbert_generator(
    shards: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    path_to_db: str,
    default_visit_id: int,
    meds_to_cehrbert_conversion_type: MedsToCehrBertConversionType,
) -> CehrBertPatient:
    conversion = get_meds_to_cehrbert_conversion_cls(
        meds_to_cehrbert_conversion_type, default_visit_id=default_visit_id
    )
    with meds_reader.SubjectDatabase(path_to_db) as patient_database:
        for shard in shards:
            for patient_id, prediction_time, label in shard:
                patient = patient_database[patient_id]
                yield convert_one_patient(patient, conversion, prediction_time, label)


def _create_cehrbert_data_from_meds(
    data_args: DataTrainingArguments,
    split: str,
    default_visit_id: int = 1,
    is_pretraining: bool = True,
):
    assert split in ["held_out", "train", "tuning"]
    batches = []
    if data_args.cohort_folder:
        # Load the entire cohort
        cohort = pd.read_parquet(os.path.expanduser(data_args.cohort_folder))
        patient_split = get_subject_split(os.path.expanduser(data_args.data_folder))
        subject_ids = patient_split[split]
        cohort_split = cohort[cohort.subject_id.isin(subject_ids)]
        for cohort_row in cohort_split.itertuples():
            subject_id = cohort_row.subject_id
            prediction_time = cohort_row.prediction_time
            label = int(cohort_row.boolean_value)
            batches.append((subject_id, prediction_time, label))
    else:
        patient_split = get_subject_split(os.path.expanduser(data_args.data_folder))
        for subject_id in patient_split[split]:
            batches.append((subject_id, None, None))

    split_batches = np.array_split(np.asarray(batches), data_args.preprocessing_num_workers)
    batch_func = functools.partial(
        _meds_to_cehrbert_generator,
        path_to_db=os.path.expanduser(data_args.data_folder),
        default_visit_id=default_visit_id,
        meds_to_cehrbert_conversion_type=data_args.meds_to_cehrbert_conversion_type,
    )
    dataset = Dataset.from_generator(
        batch_func,
        gen_kwargs={
            "shards": split_batches,
        },
        num_proc=(data_args.preprocessing_num_workers if not data_args.streaming else None),
        writer_batch_size=data_args.preprocessing_batch_size,
        streaming=data_args.streaming,
    )
    # Convert the CehrBertPatient to CehrBert data inputs
    dataset = apply_cehrbert_dataset_mapping(
        dataset,
        MedToCehrBertDatasetMapping(data_args, is_pretraining),
        num_proc=data_args.preprocessing_num_workers,
        batch_size=data_args.preprocessing_batch_size,
        streaming=data_args.streaming,
    )
    return dataset
