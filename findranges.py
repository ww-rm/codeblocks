
def findranges(values):
    """Find ranges in given values.

    Args:
        values: A iterable collection of some hashable values, and values must be comparable.

    Returns:
        ranges: A dict of (value, ranges) pairs.

    Examples:
        >>> from findranges import findranges
        >>> values = [1, 1, 1, 1, None, None, None, (9, 99), (9, 99), 1, "end"]
        >>> findranges(values)
        {1: [[0, 4], [9, 10]], None: [[4, 7]], (9, 99): [[7, 9]], 'end': [[10, 11]]}
    """

    ranges = {}
    last_v = None
    last_v_exist = False
    start_i = -1
    for i, v in enumerate(values):
        # at begining
        if not last_v_exist:
            last_v_exist = True
            last_v = v
            start_i = i
            continue

        # a new range
        if v != last_v:
            if last_v not in ranges:
                v_ranges = ranges[last_v] = []
            else:
                v_ranges = ranges[last_v]
            v_ranges.append([start_i, i])
            start_i = i

        last_v = v

    # if no values
    if not last_v_exist:
        return {}

    # process last range
    if last_v not in ranges:
        v_ranges = ranges[last_v] = []
    else:
        v_ranges = ranges[last_v]
    v_ranges.append([start_i, i + 1])

    return ranges


if __name__ == "__main__":
    import doctest
    doctest.testmod()
