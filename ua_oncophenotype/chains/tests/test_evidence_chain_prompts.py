import re

from ..evidence_chain_prompts import (
    _EVIDENCE_COT_TEMPLATE_FMT,
    _truncate_from,
    get_evidence_cot_response_pattern,
)

answer_options_short = "Yes/No/Unsure"

examples = [
    {
        "answer": "Yes",
        "evidence": "The pleural effusion on the left has increased in extent.",
    },
    {"answer": "No", "evidence": "Pleural effusions remain\nstable."},
]


def test_evidence_cot_parse():
    # make sure the regex parsing of evidence and response works correctly
    response_fmt = _truncate_from(_EVIDENCE_COT_TEMPLATE_FMT, "{{evidence}}")
    response_fmt = response_fmt.format(answer_options_short=answer_options_short)
    response_pattern = get_evidence_cot_response_pattern()
    for example in examples:
        sample_out = response_fmt.format(
            evidence=example["evidence"],
            answer_options_short=answer_options_short,
            answer=example["answer"],
        )
        m = re.match(response_pattern, sample_out, re.DOTALL)
        assert m.group(1) == example["evidence"]
        assert m.group(2) == answer_options_short
        assert m.group(3) == example["answer"]


if __name__ == "__main__":
    test_evidence_cot_parse()
