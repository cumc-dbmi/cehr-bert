import unittest
from collections import namedtuple
from datetime import datetime

from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules import MedsToCehrbertOMOP
from cehrbert.data_generators.hf_data_generator.patient_block import (
    PatientBlock,
    PatientDemographics,
    omop_meds_generate_demographics_and_patient_blocks,
)

# Mocking meds_reader.Subject and Event
MockSubject = namedtuple("Subject", ["events"])
MockEvent = namedtuple("Event", ["code", "time", "visit_id"])


class TestOmopMedsGenerateDemographicsAndPatientBlocks(unittest.TestCase):
    def setUp(self):
        """Setup mock data for the tests."""
        # Create a MedsToCehrBertConversion object (you might need to mock some of its methods)
        self.conversion = MedsToCehrbertOMOP()

        # Mocked patient events: birth date, race, gender, and a few medical events
        self.patient = MockSubject(
            events=[
                MockEvent(code="MEDS_BIRTH", time=datetime(1985, 1, 1), visit_id=None),
                MockEvent(code="RACE_WHITE", time=None, visit_id=None),
                MockEvent(code="GENDER_MALE", time=None, visit_id=None),
                MockEvent(code="ETHNICITY_NON_HISPANIC", time=None, visit_id=None),
                MockEvent(code="VS/IP", time=datetime(2021, 5, 5, 10, 0), visit_id=1),
                MockEvent(code="EVENT_2", time=datetime(2021, 5, 5, 15, 0), visit_id=1),
                MockEvent(code="EVENT_3", time=datetime(2021, 5, 5, 16, 0), visit_id=None),
                MockEvent(code="DISCHARGE", time=datetime(2021, 5, 5, 20, 0), visit_id=1),
                MockEvent(code="EVENT_4", time=datetime(2021, 5, 20, 20, 0), visit_id=None),
                MockEvent(code="VS/OP", time=datetime(2021, 6, 5, 10, 0), visit_id=2),
                MockEvent(code="EVENT_5", time=datetime(2021, 6, 5, 15, 0), visit_id=2),
            ]
        )

        # Expected demographics output
        self.expected_demographics = PatientDemographics(
            birth_datetime=datetime(1985, 1, 1),
            race="RACE_WHITE",
            gender="GENDER_MALE",
            ethnicity="ETHNICITY_NON_HISPANIC",
        )

        # Expected patient blocks
        self.expected_patient_blocks = [
            PatientBlock(
                events=[
                    MockEvent(code="VS/IP", time=datetime(2021, 5, 5, 10, 0), visit_id=1),
                    MockEvent(code="EVENT_2", time=datetime(2021, 5, 5, 15, 0), visit_id=1),
                    MockEvent(code="EVENT_3", time=datetime(2021, 5, 5, 16, 0), visit_id=None),
                    MockEvent(code="DISCHARGE", time=datetime(2021, 5, 5, 20, 0), visit_id=1),
                ],
                visit_id=1,
                conversion=self.conversion,
            ),
            PatientBlock(
                events=[
                    MockEvent(code="EVENT_4", time=datetime(2021, 5, 20, 20, 0), visit_id=None),
                ],
                visit_id=3,
                conversion=self.conversion,
            ),
            PatientBlock(
                events=[
                    MockEvent(code="VS/OP", time=datetime(2021, 6, 5, 10, 0), visit_id=2),
                    MockEvent(code="EVENT_5", time=datetime(2021, 6, 5, 15, 0), visit_id=2),
                ],
                visit_id=2,
                conversion=self.conversion,
            ),
        ]

    def test_generate_demographics_and_patient_blocks(self):
        """Test the function that generates demographics and patient blocks."""
        # Run the function
        demographics, patient_blocks = omop_meds_generate_demographics_and_patient_blocks(self.patient, self.conversion)

        # Check if the demographics match the expected output
        self.assertEqual(demographics.birth_datetime, self.expected_demographics.birth_datetime)
        self.assertEqual(demographics.race, self.expected_demographics.race)
        self.assertEqual(demographics.gender, self.expected_demographics.gender)
        self.assertEqual(demographics.ethnicity, self.expected_demographics.ethnicity)

        # Check if the patient blocks match the expected output
        self.assertEqual(len(patient_blocks), len(self.expected_patient_blocks))
        for i in range(len(self.expected_patient_blocks)):
            self.assertEqual(patient_blocks[i].visit_id, self.expected_patient_blocks[i].visit_id)
            self.assertListEqual(patient_blocks[i].events, self.expected_patient_blocks[i].events)
            self.assertEqual(patient_blocks[i].conversion, self.expected_patient_blocks[i].conversion)


if __name__ == "__main__":
    unittest.main()
