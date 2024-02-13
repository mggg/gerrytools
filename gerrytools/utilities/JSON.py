import json
from typing import Any

from pydantic import BaseModel, field_validator


class JSONtoObject(BaseModel):
    """
    Plan specification models. To better work with multiple plans at once, this
    plan specification allows users to specify information which should remain
    consistent across all operations; for example, the `column` field should be
    the name of the column in which the corresponding plan's assignment is
    stored across all data products.
    """

    column: str
    """
    Column on all data products which contains district assignment information
    for this districting plan.
    """

    locator: str
    """
    File/directory name for all resources which contain information about this
    districting plan.
    """

    title: Any = None
    """
    Official title of the plan.
    """

    type: str = None
    """
    The "type" of plan; could denote a party affiliation, a chamber, whatever.
    """

    @field_validator("column")
    def _validate_column(cls, column):
        """
        Validates the specified column name. For most of our purposes, we cannot
        include spaces, hyphens, percent signs, and other commonly-used special
        characters. Furthermore, because these plans are often saved in
        shapefiles, which have a 10-character column name limit, we force column
        names to be 10 or fewer characters.
        """
        # Check for column length.
        if len(column) > 10:
            raise ValueError(f"Column name {column} exceeds 10-character limit.")

        # Check for illegal characters.
        illegal = {"/", "%", "-", "–", "—", " "}

        for c in illegal:
            if c in column:
                raise ValueError(f"Character {c} cannot be in column name.")

        return column


def jsonify(location) -> list:
    """
    Reads in JSON data and creates a Python object out of it. If the JSON data
    read in is a list of JSON objects, a list of Python objects are returned.

    Args:
        location (string): Filepath.

    Returns:
        A list of pydantic dot-notation-accesible objects, which should contain
        information about districting plans.
    """
    with open(location) as r:
        data = json.load(r)

        # First, check whether `data` is a list; if it is, deal with all the
        # plans individually and return the list.
        if isinstance(data, list):
            return [
                JSONtoObject(
                    column=p["column"],
                    locator=p["locator"],
                    title=p["title"] if p.get("title", False) else None,
                    type=p["type"] if p.get("type", False) else None,
                )
                for p in data
            ]

        # Otherwise, return a singleton list with a single plan.
        return [
            JSONtoObject(
                column=data["column"],
                locator=data["locator"],
                title=data["title"] if data.get("title", False) else None,
            )
        ]
