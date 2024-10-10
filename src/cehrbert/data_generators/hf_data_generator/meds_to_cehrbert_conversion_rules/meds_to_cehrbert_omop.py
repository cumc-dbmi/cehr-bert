from typing import List

from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    EventConversionRule,
    MedsToCehrBertConversion,
)


class MedsToCehrbertOMOP(MedsToCehrBertConversion):

    def _create_ed_admission_matching_rules(self) -> List[str]:
        return ["Visit/ER"]

    def _create_admission_matching_rules(self) -> List[str]:
        return ["Visit/IP", "Visit/ERIP", "CMS Place of Service/51", "CMS Place of Service/61"]

    def _create_discharge_matching_rules(self) -> List[str]:
        return [
            "PCORNet/Generic-NI",
            "CMS Place of Service/12",
            "SNOMED/371827001",
            "CMS Place of Service/21",
            "NUCC/261Q00000X",
            "CMS Place of Service/31",
            "SNOMED/397709008",
            "Medicare Specialty/A4",
            "SNOMED/225928004",
            "CMS Place of Service/34",
            "CMS Place of Service/61",
            "CMS Place of Service/51",
            "CMS Place of Service/23",
            "PCORNet/Generic-OT",
            "CMS Place of Service/27",
            "CMS Place of Service/24",
            "CMS Place of Service/09",
            "CMS Place of Service/33",
            "SNOMED/34596002",
            "CMS Place of Service/25",
            "CMS Place of Service/32",
            "CMS Place of Service/20",
        ]

    def _create_text_event_to_numeric_event_rules(self) -> List[EventConversionRule]:
        return []
