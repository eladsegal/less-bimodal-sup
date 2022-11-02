from collections.abc import Mapping, Sequence


def get_dot(obj, dot_notation):
    exists = False
    steps = dot_notation.split(".")
    current_obj = obj
    for i, step in enumerate(steps):
        if isinstance(current_obj, Sequence) and not isinstance(current_obj, str):
            try:
                step = int(step)
            except Exception as e:
                raise ValueError(f"Trying to access a list with a non-integer index: {step}")
            if len(current_obj) > int(step):
                exists = True
                current_obj = current_obj[int(step)]
        elif isinstance(current_obj, Mapping):
            if step in current_obj:
                exists = True
                current_obj = current_obj[step]
        else:
            current_dot_notation = ".".join(steps[:i])
            raise Exception(f"Unsupported type {type(obj)} in `{current_dot_notation}`")

    return exists, (current_obj if exists else None)


def set_dot(obj, dot_notation, value, force=False):
    steps = dot_notation.split(".")

    if force:
        current_obj = obj
        for i in range(len(steps) - 1):
            exists, x = get_dot(current_obj, steps[i])
            if exists:
                current_obj = x
            else:
                current_obj[steps[i]] = {}
                current_obj = current_obj[steps[i]]
        penultimate_obj = current_obj
    else:
        if len(steps[:-1]) > 0:
            exists, penultimate_obj = get_dot(obj, ".".join(steps[:-1]))
        else:
            exists = True
            penultimate_obj = obj

        if not exists:
            raise Exception(f"`{'.'.join(steps[:-1])}` does not exist in")

    if isinstance(penultimate_obj, Sequence) and not isinstance(penultimate_obj, str):
        try:
            step = int(steps[-1])
        except Exception as e:
            raise ValueError(f"Trying to access a list with a non-integer index: {steps[-1]}")
    elif isinstance(penultimate_obj, Mapping):
        step = steps[-1]
    else:
        raise Exception(f"Unsupported type {type(obj)} in `{dot_notation}`")
    penultimate_obj[step] = value
