import datetime as dt
import logging
from typing import List
from uuid import UUID

from pydantic import BaseModel
from rwd_llm.dtypes.dtypes import BaseObject, ClinicalNote

logger = logging.getLogger(__name__)


class NoteFormatterBase:
    """Class to take a list of notes and format it as useful prompt input"""

    @classmethod
    def format(cls, notes: List[ClinicalNote]) -> str:
        raise NotImplementedError()


BASIC_NOTE_FMT = """
Note Type: {note_type}
Date: {note_date}
----------
{note_text}
----------
"""

NOTE_SEP = "\n\n----------------------------------------\n\n"


class BasicNoteFormatter(NoteFormatterBase):
    """Class to take a list of notes and format it as useful prompt input"""

    @classmethod
    def format(cls, notes: List[ClinicalNote]) -> str:
        note_strings = [
            BASIC_NOTE_FMT.format(
                note_type=note.type, note_date=note.date, note_text=note.text
            )
            for note in notes
        ]
        return NOTE_SEP.join(note_strings)


class NoteCombiner:
    """Class to take a list of notes and format it as useful prompt input"""

    @classmethod
    def format(cls, notes: List[ClinicalNote]) -> str:
        note_strings = [
            BASIC_NOTE_FMT.format(
                note_type=note.type, note_date=note.date, note_text=note.text
            )
            for note in notes
        ]
        return NOTE_SEP.join(note_strings)


def force_to_json(json_ob):
    """
    Make sure json_ob is json-serializable, including rwd_llm BaseObject subclasses. If
    something can't be serialized as json, just fall back to representing it as a 'repr'
    string.
    """

    if isinstance(json_ob, (str, float, int, bool)):
        return json_ob
    elif isinstance(json_ob, dict):
        return {key: force_to_json(value) for key, value in json_ob.items()}
    elif isinstance(json_ob, (list, tuple, set)):
        return [force_to_json(sub_ob) for sub_ob in json_ob]
    elif isinstance(json_ob, dt.datetime):
        return json_ob.isoformat(timespec="seconds")
    elif isinstance(json_ob, UUID):
        return str(json_ob)
    elif isinstance(json_ob, BaseObject):
        return force_to_json(json_ob.to_dict())
    elif isinstance(json_ob, BaseModel):
        return force_to_json(json_ob.model_dump())
    elif json_ob is None:
        return None
    else:
        logger.error(f"Cannot sanitize {json_ob} of type {type(json_ob)}.")
        # just replace with a string representation, unless repr is very large
        str_repr = repr(json_ob)
        if len(str_repr) > 1000:
            return f"not serializing {str(type(json_ob))} of size {len(str_repr)}"
