from typing import Dict, List, Union

import numpy as np


def input_maker(**kwargs) -> List[Dict[str, Union[int, float, np.array]]]:
    """
    easily make an input arguments list for `Runner.run()`

    for example:

    .. code-block:: python

        input_maker(a=a, b=b, length=(arr_length, "int"), out=out)

    then got a list of dict

    .. code-block::

        [
            {"name": "a", "value": a},
            {"name": "b", "value": b},
            {"name": "length", "value": arr_length, "type": "int"},
            {"name": "out", "value": out},
        ]

    :param kwargs: key value arguments,
                    if value is tuple, should be (value, type)
    :return: input arg list
    :rtype: List[Dict[str, Any]]
    """
    input = []
    for key, value in kwargs.items():
        if isinstance(value, tuple):
            v, t = value
            input.append({"name": key, "value": v, "type": t})
        else:
            input.append({"name": key, "value": value})
    return input
