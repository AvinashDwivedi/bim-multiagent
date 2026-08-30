from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def parse_strict_decimal_token(
    value: str,
) -> tuple[Decimal | None, str, str | None]:
    """Parse one locale-independent scalar token without guessing separators.

    A single separator followed by exactly three digits is ambiguous without
    locale metadata (``1,234`` and ``1.234`` can each mean either a decimal or
    a grouped integer), so it is deliberately rejected. Mixed separators are
    accepted only when the last separator is decimal and all preceding groups
    are valid thousands groups.
    """

    token = str(value).strip()
    # Compatibility folding can turn a superscript/circled sequence into a
    # different ASCII number (for example ``10²`` -> ``102``).  Numeric
    # evidence therefore accepts only the explicit ASCII scalar grammar.
    if any(ord(character) > 127 for character in token):
        return None, "rejected", "non_ascii_number"
    if re.fullmatch(
        r"[-+]?(?:[0-9]+(?:[.,][0-9]+)*|[.,][0-9]+)(?:[eE][-+]?[0-9]+)?",
        token,
    ) is None:
        return None, "rejected", "invalid_number"

    exponent = ""
    mantissa = token
    exponent_match = re.fullmatch(r"(.+?)([eE][-+]?\d+)", token)
    if exponent_match:
        mantissa, exponent = exponent_match.groups()

    comma_count = mantissa.count(",")
    dot_count = mantissa.count(".")
    parse_format = "integer"
    normalized = mantissa
    if comma_count and dot_count:
        decimal_separator = "," if mantissa.rfind(",") > mantissa.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        integer_part, fractional_part = mantissa.rsplit(decimal_separator, 1)
        groups = integer_part.lstrip("+-").split(grouping_separator)
        if (
            re.fullmatch(r"[0-9]+", fractional_part) is None
            or not groups
            or not 1 <= len(groups[0]) <= 3
            or any(
                len(group) != 3 or re.fullmatch(r"[0-9]{3}", group) is None
                for group in groups[1:]
            )
        ):
            return None, "rejected", "invalid_grouping"
        sign = "-" if integer_part.startswith("-") else "+" if integer_part.startswith("+") else ""
        normalized = sign + "".join(groups) + "." + fractional_part
        parse_format = "grouped_decimal"
    elif comma_count or dot_count:
        separator = "," if comma_count else "."
        count = comma_count or dot_count
        signless = mantissa.lstrip("+-")
        groups = signless.split(separator)
        if count > 1:
            if (
                not 1 <= len(groups[0]) <= 3
                or any(
                    len(group) != 3 or re.fullmatch(r"[0-9]{3}", group) is None
                    for group in groups[1:]
                )
            ):
                return None, "rejected", "invalid_grouping"
            sign = "-" if mantissa.startswith("-") else "+" if mantissa.startswith("+") else ""
            normalized = sign + "".join(groups)
            parse_format = "grouped_integer"
        else:
            integer_part, fractional_part = groups
            if len(fractional_part) == 3 and 1 <= len(integer_part) <= 3:
                return None, "ambiguous", "single_separator_three_digits"
            normalized = mantissa.replace(separator, ".")
            parse_format = "decimal_comma" if separator == "," else "decimal_point"

    try:
        return Decimal(normalized + exponent), "parsed", parse_format
    except InvalidOperation:
        return None, "rejected", "invalid_number"
