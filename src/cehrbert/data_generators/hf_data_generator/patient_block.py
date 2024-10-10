import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import meds_reader

from cehrbert.data_generators.hf_data_generator import (
    DEFAULT_ED_CONCEPT_ID,
    DEFAULT_INPATIENT_CONCEPT_ID,
    DEFAULT_OUTPATIENT_CONCEPT_ID,
)
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import birth_codes
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules import (
    MedsToBertMimic4,
    MedsToCehrBertConversion,
    MedsToCehrbertOMOP,
)
from cehrbert.med_extension.schema_extension import Event


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


def generate_demographics_and_patient_blocks(
    conversion: MedsToCehrBertConversion,
    patient: meds_reader.Subject,
    prediction_time: datetime = None,
) -> Tuple[PatientDemographics, List[PatientBlock]]:
    if isinstance(conversion, MedsToBertMimic4):
        return mimic_meds_generate_demographics_and_patient_blocks(
            patient, conversion, prediction_time, conversion.default_visit_id
        )
    elif isinstance(conversion, MedsToCehrbertOMOP):
        return omop_meds_generate_demographics_and_patient_blocks(patient, conversion, prediction_time)
    else:
        raise RuntimeError(f"Unrecognized conversion: {conversion}")


def omop_meds_generate_demographics_and_patient_blocks(
    patient: meds_reader.Subject, conversion: MedsToCehrBertConversion, prediction_time: datetime = None
) -> Tuple[PatientDemographics, List[PatientBlock]]:
    birth_datetime = None
    race = None
    gender = None
    ethnicity = None

    visit_events = defaultdict(list)
    unlinked_event_mapping = defaultdict(list)
    for e in patient.events:
        # This indicates demographics features
        if e.code in birth_codes:
            birth_datetime = e.time
        elif e.code.upper().startswith("RACE"):
            race = e.code
        elif e.code.upper().startswith("GENDER"):
            gender = e.code
        elif e.code.upper().startswith("ETHNICITY"):
            ethnicity = e.code
        elif e.time is not None:
            # Skip out of the loop if the events' time stamps are beyond the prediction time
            if prediction_time is not None:
                if e.time > prediction_time:
                    break

            if getattr(e, "visit_id", None):
                visit_id = e.visit_id
                visit_events[visit_id].append(e)
            else:
                unlinked_event_mapping[e.time.strftime("%Y-%m-%d")].append(e)

    patient_block_mapping = {
        visit_id: PatientBlock(events=events, visit_id=visit_id, conversion=conversion)
        for visit_id, events in visit_events.items()
    }

    # Try to connect the unlinked events to existing visits
    for current_date_str in list(unlinked_event_mapping.keys()):
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
        for visit_id, patient_block in patient_block_mapping.items():
            if patient_block.min_time.date() <= current_date <= patient_block.max_time.date():
                patient_block.events.extend(unlinked_event_mapping.pop(current_date_str, []))
                # Need to sort the events if we insert new events to the patient block
                patient_block.events = sorted(patient_block.events, key=lambda _: _.time)
                break

    max_visit_id = max(patient_block_mapping.keys()) + 1 if len(patient_block_mapping) > 0 else 1
    for events in unlinked_event_mapping.values():
        patient_block_mapping[max_visit_id] = PatientBlock(events, max_visit_id, conversion)
        max_visit_id += 1

    patient_blocks = list(patient_block_mapping.values())
    demographics = PatientDemographics(birth_datetime=birth_datetime, race=race, gender=gender, ethnicity=ethnicity)

    # If there are unlinked events, we need to add them as new patient blocks, therefore we need to re-order the patient block
    if len(unlinked_event_mapping) > 0:
        patient_blocks = sorted(patient_block_mapping.values(), key=lambda block: block.min_time)

    return demographics, patient_blocks


def mimic_meds_generate_demographics_and_patient_blocks(
    patient: meds_reader.Subject,
    conversion: MedsToCehrBertConversion,
    prediction_time: datetime = None,
    default_visit_id: int = 1,
) -> Tuple[PatientDemographics, List[PatientBlock]]:
    birth_datetime = None
    race = None
    gender = None
    ethnicity = None

    visit_id = default_visit_id
    current_date = None
    events_for_current_date = []
    patient_blocks = []
    for e in patient.events:

        # Skip out of the loop if the events' time stamps are beyond the prediction time
        if prediction_time is not None and e.time is not None:
            if e.time > prediction_time:
                break

        # This indicates demographics features
        if e.code in birth_codes:
            birth_datetime = e.time
        elif e.code.upper().startswith("RACE"):
            race = e.code
        elif e.code.upper().startswith("GENDER"):
            gender = e.code
        elif e.code.upper().startswith("ETHNICITY"):
            ethnicity = e.code
        elif e.time is not None:
            if not current_date:
                current_date = e.time

            if current_date.date() == e.time.date():
                events_for_current_date.append(e)
            else:
                patient_blocks.append(PatientBlock(events_for_current_date, visit_id, conversion))
                events_for_current_date = [e]
                current_date = e.time
                visit_id += 1

    if events_for_current_date:
        patient_blocks.append(PatientBlock(events_for_current_date, visit_id, conversion))

    admit_discharge_pairs = []
    active_ed_index = None
    active_admission_index = None
    # |ED|24-hours|Admission| ... |Discharge| -> ED will be merged into the admission (within 24 hours)
    # |ED|25-hours|Admission| ... |Discharge| -> ED will NOT be merged into the admission
    # |Admission|ED| ... |Discharge| -> ED will be merged into the admission
    # |Admission|Admission|ED| ... |Discharge|
    #   -> The first admission will be ignored and turned into a separate visit
    #   -> The second Admission and ED will be merged
    for i, patient_block in enumerate(patient_blocks):
        # Keep track of the ED block when there is no on-going admission
        if patient_block.has_ed_admission and active_admission_index is None:
            active_ed_index = i
        # Keep track of the admission block
        if patient_block.has_admission:
            # If the ED event has occurred, we need to check the time difference between
            # the ED event and the subsequent hospital admission
            if active_ed_index is not None:

                hour_diff = (patient_block.min_time - patient_blocks[active_ed_index].max_time).total_seconds() / 3600
                # If the time difference between the ed and admission is leq 24 hours,
                # we consider ED to be part of the visits
                if hour_diff <= 24 or active_ed_index == i:
                    active_admission_index = active_ed_index
                    active_ed_index = None
            else:
                active_admission_index = i

        if patient_block.has_discharge:
            if active_admission_index is not None:
                admit_discharge_pairs.append((active_admission_index, i))
            # When the patient is discharged from the hospital, we assume the admission and ED should end
            active_admission_index = None
            active_ed_index = None

        # Check the last block of the patient history to see whether the admission is partial
        if i == len(patient_blocks) - 1:
            # This indicates an ongoing (incomplete) inpatient visit,
            # this is a common pattern for inpatient visit prediction problems,
            # where the data from the first 24-48 hours after the admission
            # are used to predict something about the admission
            if active_admission_index is not None and prediction_time is not None:
                admit_discharge_pairs.append((active_admission_index, i))

    # Update visit_id for the admission blocks
    for admit_index, discharge_index in admit_discharge_pairs:
        admission_block = patient_blocks[admit_index]
        discharge_block = patient_blocks[discharge_index]
        visit_id = admission_block.visit_id
        for i in range(admit_index, discharge_index + 1):
            patient_blocks[i].visit_id = visit_id
            patient_blocks[i].visit_type = DEFAULT_INPATIENT_CONCEPT_ID
        # There could be events that occur after the discharge, which are considered as part of the visit
        # we need to check if the time stamp of the next block is within 12 hours
        if discharge_index + 1 < len(patient_blocks):
            next_block = patient_blocks[discharge_index + 1]
            hour_diff = (next_block.min_time - discharge_block.max_time).total_seconds() / 3600
            assert hour_diff >= 0, (
                f"next_block.min_time: {next_block.min_time} "
                f"must be GE discharge_block.max_time: {discharge_block.max_time}"
            )
            if hour_diff <= 12:
                next_block.visit_id = visit_id
                next_block.visit_type = DEFAULT_INPATIENT_CONCEPT_ID

    demographics = PatientDemographics(birth_datetime=birth_datetime, race=race, gender=gender, ethnicity=ethnicity)
    return demographics, patient_blocks
