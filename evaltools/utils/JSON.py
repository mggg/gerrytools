
import json
from typing import List, Union

class JSON(object):
    """
    A class which represents JSON data as a Python object rather than a dict. If
    the JSON object is a list, returns a list of JSON-ified objects.
    """
    def __init__(self, data):
        """
        Args:
            data (string): Data to jsonify.
        """
        for k, v in data.items(): self.__dict__[k] = v

def objectify(location) -> Union[List[JSON],JSON]:
    """
    Reads in JSON data and creates a Python object out of it. If the JSON data
    read in is a list of JSON objects, a list of Python objects are returned.

    Args:
        location (string): Filepath.

    Returns:
        A dot-notation-accessible object or list of dot-notation-accessible
        objects.
    """
    with open(location) as r:
        data = json.load(r)

        if type(data) is list: return [JSON(p) for p in data]
        return JSON(data)