def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result
