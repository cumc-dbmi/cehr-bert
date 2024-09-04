import re
from typing import List

from data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    MedsToCehrBertConversion, EventConversionRule
)


class MedsToBertMimic4(MedsToCehrBertConversion):

    def get_ed_admission_matching_rules(self) -> List[str]:
        return ["ED_REGISTRATION//", "TRANSFER_TO//ED"]

    def get_admission_matching_rules(self) -> List[str]:
        return ["HOSPITAL_ADMISSION//"]

    def get_discharge_matching_rules(self) -> List[str]:
        return ["HOSPITAL_DISCHARGE//"]

    def _text_event_to_numeric_events(self) -> List[EventConversionRule]:
        blood_pressure_codes = [
            "Blood Pressure",
            "Blood Pressure Lying",
            "Blood Pressure Sitting",
            "Blood Pressure Standing (1 min)",
            "Blood Pressure Standing (3 mins)"
        ]
        blood_pressure_rules = [
            EventConversionRule(
                code=code,
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=[f"Systolic {code}", f"Diastolic {code}"]
            )
            for code in blood_pressure_codes
        ]
        height_weight_codes = ["Weight (Lbs)", "Height (Inches), BMI (kg/m2)"]
        height_weight_rules = [
            EventConversionRule(
                code=code,
                parsing_pattern=re.compile(r"(\d+)"),
                mapped_event_labels=[code]
            )
            for code in height_weight_codes
        ]
        ventilation_rate_rules = [
            EventConversionRule(
                code="LAB//50827//UNK",
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=["LAB//50827//UNK//1", "LAB//50827//UNK//2"]
            )
        ]
        return blood_pressure_rules + height_weight_rules + ventilation_rate_rules

    def get_open_ended_event_codes(self) -> List[str]:
        return ["LAB//220001//UNK"]
