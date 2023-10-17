import copy
import json
import os
from typing import Dict, List, Optional

from langchain.schema import BaseMemory, Document

from ..utils import get_by_path
from .mapping_chain import MappingChain
from .structured_summary import ParsedSummaryEvidence, Summary


class SummaryDeserializer:
    def __call__(self, summary_list: List[Dict]):
        return [
            parse_summarized_doc_from_dict(summary_dict)
            for summary_dict in summary_list
        ]


def parse_summarized_doc_from_dict(summarized_doc_dict: dict) -> Document:
    summarized_doc = Document(**summarized_doc_dict)
    structured_summary_dict = summarized_doc.metadata["summary"]
    summarized_doc.metadata["summary"] = parse_summary_from_dict(
        structured_summary_dict
    )
    return summarized_doc


def parse_summary_from_dict(summary_dict: dict) -> Summary:
    """Convert a summary dict to a Summary object. We need to manually convert
    ParsedSummaryEvidence"""
    summary_dict = copy.deepcopy(summary_dict)
    for finding_dict in summary_dict["findings"]:
        evidence = []
        for evidence_dict in finding_dict["evidence"]:
            evidence.append(ParsedSummaryEvidence(**evidence_dict))
        finding_dict["evidence"] = evidence
    return Summary(**summary_dict)


def mapping_chain_from_traces(
    trace_dir: str,
    key_trace_path: str,
    value_trace_path: str,
    input_key: str,
    output_key: str,
    memory: Optional[BaseMemory] = None,
) -> MappingChain:
    """Given a path to a trace directory and the sub-paths to the key and value,
    create a MappingChain from key to value for each trace."""
    mapping = {}
    for path in os.listdir(os.path.join(trace_dir, "traces")):
        try:
            if not path.endswith(".json"):
                continue
            with open(os.path.join(trace_dir, "traces", path), "r") as f:
                trace = json.load(f)
            try:
                the_key = get_by_path(trace, key_trace_path)
            except Exception as e:
                print(f"Failed to parse {key_trace_path} from {path}: {e}")
                continue
            try:
                the_val = get_by_path(trace, value_trace_path)
            except Exception as e:
                print(f"Failed to parse {key_trace_path} from {path}: {e}")
                continue
            mapping[the_key] = the_val
        except Exception as e:
            print(f"Failed to parse {path}: {e}")
    return MappingChain(
        mapping=mapping,
        memory=memory,
        input_key=input_key,
        output_key=output_key,
    )


def generate_structured_summary_mapping_chain(
    trace_dir: str,
    summaries_path: str,
    id_path: str,
    input_key: str = "patient_id",
    output_key: str = "patient_history",
    memory: Optional[BaseMemory] = None,
) -> MappingChain:
    """Given a path to a trace directory and the sub-path to the summaries, inject the
    summaries based on ID (also given by path)"""
    id_to_summaries_mapping = {}
    for path in os.listdir(os.path.join(trace_dir, "traces")):
        try:
            if not path.endswith(".json"):
                continue
            with open(os.path.join(trace_dir, "traces", path), "r") as f:
                trace = json.load(f)
            the_id = get_by_path(trace, id_path)
            summarized_docs = get_by_path(trace, summaries_path)
            parsed_summarized_docs = [
                parse_summarized_doc_from_dict(summarized_doc_dict)
                for summarized_doc_dict in summarized_docs
            ]
            id_to_summaries_mapping[the_id] = parsed_summarized_docs
        except Exception as e:
            print(f"Failed to parse {summaries_path} from {path}: {e}")
    return MappingChain(
        mapping=id_to_summaries_mapping,
        memory=memory,
        input_key=input_key,
        output_key=output_key,
    )
