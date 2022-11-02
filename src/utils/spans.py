from typing import List, Tuple, Sequence


def extract_span_string_from_origin_texts(
    span: Tuple[int, int],
    origin_texts: Sequence[str],
    offset_mapping: List[Tuple[int, int]],
    sequence_ids: List[int],
):
    if span[0] == -1 or span[1] == -1:
        return ""
    if span[0] >= len(offset_mapping):
        return ""
    if span[1] >= len(offset_mapping):
        span = (span[0], len(offset_mapping) - 1)

    sequence_ranges = _sequence_ids_to_ranges(sequence_ids)
    start_span_origin_index = None
    end_span_origin_index = None
    for i in range(len(origin_texts)):
        sequence_range = sequence_ranges[i]

        if sequence_range[0] <= span[0] <= sequence_range[1]:
            start_span_origin_index = i
            if end_span_origin_index is not None:
                break
        if sequence_range[0] <= span[1] <= sequence_range[1]:
            end_span_origin_index = i
            if start_span_origin_index is not None:
                break

    if start_span_origin_index is None:
        if span[0] > sequence_ranges[-1][1]:
            return ""
        for i in range(len(sequence_ranges)):
            if span[0] < sequence_ranges[i][0]:
                start_span_origin_index = i
                break

    if end_span_origin_index is None:
        if span[1] < sequence_ranges[0][0]:
            return ""
        for i in range(len(sequence_ranges) - 1, -1, -1):
            if span[1] > sequence_ranges[i][1]:
                end_span_origin_index = i
                break

    span_string_parts = []
    for i in range(start_span_origin_index, end_span_origin_index + 1):
        character_start = offset_mapping[max(sequence_ranges[i][0], span[0])][0]
        character_end = offset_mapping[min(sequence_ranges[i][1], span[1])][1]
        span_string_parts.append(origin_texts[i][character_start:character_end])
    span_string = " ".join(span_string_parts)

    return span_string


def get_token_span(
    offset_mapping: List[Tuple[int, int]],
    special_tokens_mask: List[int],
    start_offset: int,
    end_offset: int,
    sequence_index: int,
):
    sequence_range = _get_sequence_boundaries(special_tokens_mask)[sequence_index]

    if start_offset < offset_mapping[sequence_range[0]][0] or offset_mapping[sequence_range[1]][1] < end_offset:
        return None

    answer_token_indices = []
    for i, offset in enumerate(offset_mapping):
        if i < sequence_range[0] or i > sequence_range[1]:
            continue
        is_start = offset[0] <= start_offset and start_offset < offset[1]
        is_mid = start_offset <= offset[0] and offset[1] <= end_offset
        is_end = offset[0] < end_offset and end_offset <= offset[1]

        if is_start or is_mid or is_end:
            answer_token_indices.append(i)

    token_span = (answer_token_indices[0], answer_token_indices[-1])
    return token_span


def _sequence_ids_to_ranges(sequence_ids: List[int]) -> List[Tuple[int, int]]:
    sequence_ranges = []
    range_start, range_end = None, None
    for i, id_ in enumerate(sequence_ids):
        if range_start is not None and id_ != sequence_ids[range_start]:
            range_end = i - 1
            sequence_ranges.append((range_start, range_end))
            range_start = None
        if range_start is None and id_ is not None:
            range_start = i
    if range_start is not None:
        range_end = len(sequence_ids) - 1
        sequence_ranges.append((range_start, range_end))
    return sequence_ranges


def _get_sequence_boundaries(special_tokens_mask: List[int]) -> List[Tuple[int, int]]:
    """
    Returns the token index boundaries of a sequence that was encoded together with other sequences,
    by using special_tokens_mask.
    """
    boundaries = []
    start_index = None
    special_sequence = True
    for i, value in enumerate(special_tokens_mask):
        if value == 0:
            if special_sequence is True:
                start_index = i
                special_sequence = False
        elif value == 1:
            if special_sequence is False:
                boundaries.append((start_index, (i - 1)))
                special_sequence = True
    if special_sequence is False:
        boundaries.append((start_index, len(special_tokens_mask) - 1))
    return boundaries
