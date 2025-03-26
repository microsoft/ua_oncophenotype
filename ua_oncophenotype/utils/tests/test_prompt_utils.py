from ..prompt_utils import find_format_variables


def test_find_format_variables():
    # Test
    s = (
        "Hello, {name}. You have {count} new messages. These are not format variables:"
        " {{hello}} and {{world}}"
    )
    format_vars = set(find_format_variables(s))
    assert format_vars == set(["name", "count"])


if __name__ == "__main__":
    test_find_format_variables()
