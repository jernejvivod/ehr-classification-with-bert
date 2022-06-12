def process_param_spec(params_str: str) -> dict:
    """Construction dictionary from function parameters specified using the command-line interface
    with key-value pairs (val=3).

    :param params_str: string specifying the key value pairs
    :return: constructed dictionary
    """

    split_str = {kv[0]: kv[1] for exp in params_str.split() for kv in [exp.split('=')]}
    params_dict = dict()
    for k, v in split_str.items():
        try:
            params_dict[k] = int(v)
        except ValueError:
            try:
                params_dict[k] = float(v)
            except ValueError:
                params_dict[k] = v
    return params_dict
