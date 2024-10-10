import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EventConversionRule:
    """
    Represents a rule for converting an event code into corresponding event labels.

    based on a regular expression pattern.

    Attributes:
        code (str): The code associated with the event that needs to be parsed.
        parsing_pattern (re.Pattern): The regular expression pattern used to parse the event code.
        mapped_event_labels (List[str]): A list of event labels mapped to the groups
                                         in the regular expression pattern.

    Methods:
        __post_init__(): Ensures that the number of regex groups matches the number of mapped event
        labels. This method is automatically called after the object is initialized.
    """

    code: str
    parsing_pattern: re.Pattern
    mapped_event_labels: List[str]

    def __post_init__(self):
        assert self.parsing_pattern.groups == len(
            self.mapped_event_labels
        ), "The number the mapped event labels needs to match the number of groups in the regex"


class MedsToCehrBertConversion(ABC):
    """
    Abstract base class for converting medication-related text events into numeric event labels.

    for CehR-BERT models. This class provides an interface for defining matching rules for
    ED admission, general admission, discharge, and text-to-numeric event mappings.

    Attributes:
        _ed_admission_matching_rules (List[str]): Cached matching rules for identifying ED admissions.
        _admission_matching_rules (List[str]): Cached matching rules for identifying admissions.
        _discharge_matching_rules (List[str]): Cached matching rules for identifying discharges.
        _text_event_numeric_event_map (dict): Cached map of text event codes to EventConversionRule objects.

    Methods:
        _create_ed_admission_matching_rules(): Abstract method for creating ED admission matching rules.
        _create_admission_matching_rules(): Abstract method for creating admission matching rules.
        _create_discharge_matching_rules(): Abstract method for creating discharge matching rules.
        _create_text_event_to_numeric_event_rules(): Abstract method for creating text-to-numeric event rules.
        get_ed_admission_matching_rules(): Returns the ED admission matching rules.
        get_admission_matching_rules(): Returns the general admission matching rules.
        get_discharge_matching_rules(): Returns the discharge matching rules.
        get_text_event_to_numeric_events_rule(): Returns the EventConversionRule for a given code,
                                                 or None if no rule exists.
    """

    def __init__(self, **kwargs):
        """
        Initializes the MedsToCehrBertConversion class by caching the matching rules and.

        text-to-numeric event mappings, which are created by calling the respective abstract methods.
        """
        # Cache these variables once
        self._ed_admission_matching_rules = self._create_ed_admission_matching_rules()
        self._admission_matching_rules = self._create_admission_matching_rules()
        self._discharge_matching_rules = self._create_discharge_matching_rules()
        self._text_event_numeric_event_map = {r.code: r for r in self._create_text_event_to_numeric_event_rules()}

    @abstractmethod
    def _create_ed_admission_matching_rules(self) -> List[str]:
        """
        Abstract method for defining the matching rules for identifying ED admissions.

        Returns:
            List[str]: A list of rules for identifying ED admissions.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError("Must implement the matching rules for identifying the ED admission")

    @abstractmethod
    def _create_admission_matching_rules(self) -> List[str]:
        """
        Abstract method for defining the matching rules for identifying admissions.

        Returns:
            List[str]: A list of rules for identifying admissions.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError("Must implement the matching rules for identifying the admission")

    @abstractmethod
    def _create_discharge_matching_rules(self) -> List[str]:
        """
        Abstract method for defining the matching rules for identifying discharges.

        Returns:
            List[str]: A list of rules for identifying discharges.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError("Must implement the matching rules for identifying the discharge")

    @abstractmethod
    def _create_text_event_to_numeric_event_rules(self) -> List[EventConversionRule]:
        """
        Abstract method for defining the rules for mapping text events to numeric events.

        Returns:
            List[EventConversionRule]: A list of event conversion rules mapping text events
            to numeric events.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError("Must implement the event mapping rules for converting text events to numeric events")

    def get_ed_admission_matching_rules(self) -> List[str]:
        return self._ed_admission_matching_rules

    def get_admission_matching_rules(self) -> List[str]:
        return self._admission_matching_rules

    def get_discharge_matching_rules(self) -> List[str]:
        return self._discharge_matching_rules

    def get_text_event_to_numeric_events_rule(self, code) -> Optional[EventConversionRule]:
        if code in self._text_event_numeric_event_map:
            return self._text_event_numeric_event_map[code]
        return None
