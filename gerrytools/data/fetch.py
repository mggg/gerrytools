import io
import json
from datetime import datetime
from typing import List, Tuple, Union

import pandas as pd
import requests
from pydantic import BaseModel

from .URLs import csvs, ids, one


class Submission(BaseModel):
    """
    Provides a base model for data retrieved from districtr. Allows us to use
    dot notation when accessing properties rather than dict notation.
    """

    link: str
    """A districtr URL."""
    plan: dict
    """districtr plan object."""
    id: str
    """districtr identifier."""
    units: str
    """Unit identifier (e.g. `GEOID`)."""
    unitsType: str
    """Unit type (e.g. `blocks20`, `blockgroup`, etc.)"""
    tileset: str
    """Mapbox tileset URL."""
    type: str
    """Not sure."""


def tabularized(state, submissions) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns districtr submission information in a tabular format.

    Args:
        state (State): `us.State` object (e.g. `us.states.WI`).
        submissions (list): List of `Submission` objects returned from a
            call to `submissions`.

    Returns:
        Three dataframes corresponding to plan-based submissions, COI-based
        submissions, and written submissions to the provided state.

    Example:
        Prototypical example usage.

            import us
            from gerrytools.retrieve import submissions, tabularized

            # Set the state.
            state = us.states.WI

            # Retrieve the raw districtr submissions, then tabularize them.
            subs = submissions(state)
            plans, cois, written = tabularized(state, subs)

    """
    # Sort submissions. (Not sure why this is necessary? Holdover from previous
    # fetching code.)
    submissions = list(sorted(submissions, key=lambda s: s.id))

    # Categorize into three categories: plan submissions, COI submissions, and
    # written submissions (which are ignored as they don't appear in the list
    # of submissions).
    _plans = [s.dict() for s in submissions if s.type == "plan"]
    _cois = [s.dict() for s in submissions if s.type == "coi"]

    # Create preliminary dataframes so we can do safe `merge`s rather than rely
    # explicitly on sorting; this also allows us to specify a sample size if
    # we're only looking to sample a specific number of plans.
    subset_plans = pd.DataFrame.from_records(_plans)
    subset_cois = pd.DataFrame.from_records(_cois)

    # Get appropriate URLs and create dataframes.
    plans_url = csvs(state)
    cois_url = csvs(state, ptype="coi")
    written_url = csvs(state, ptype="written")

    plans = as_dataframe(plans_url)
    cois = as_dataframe(cois_url)
    writtens = as_dataframe(written_url)

    # Adjust column contents for the plan and COI dataframes.
    for universe in [plans, cois]:
        # Adjust the `link` column type and create an `id` column from it.
        universe["link"] = universe["link"].astype(str)
        universe["id"] = parse_id(universe["link"])

    # Adjust column contents for all dataframes.
    for df in [plans, cois, writtens]:
        df["datetime"] = parse_datetime(df["datetime"])

    # Add the retrieved plan data to the dataframes *if the subset dataframes
    # contain items*.
    if not subset_plans.empty:
        plans = plans.merge(subset_plans, on="id")
    else:
        plans = pd.DataFrame()
    if not subset_cois.empty:
        cois = cois.merge(subset_cois, on="id")
    else:
        cois = pd.DataFrame()

    # Drop bad columns and rename. Not sure why we have to `inplace`
    # things here, but... fine.
    for df in [plans, cois]:
        if not df.empty:
            # Remove columns we don't necessarily care about.
            for col in ["type_x", "link_x", "coalition"]:
                if col in list(df):
                    df.drop(col, axis=1, inplace=True)

            # Rename the columns we do care about.
            df.rename({"type_y": "type", "link_y": "link"}, axis=1, inplace=True)

    return plans, cois, writtens


def submissions(state, sample=None) -> List[Submission]:
    """
    Retrieves raw districtr objects; this includes both plan- and COI-based
    submissions.

    Args:
        state (State): `us.State` object (e.g. `us.states.WI`).
        sample (int, optional): The number of sample plans to retrieve.

    Returns:
        A list of `Submissions`, either to be interpreted raw or tabularized.
    """
    # Get the appropriate URL and send the request. Made some basic ASCII
    # art with the second three variable names... it's like the request
    # is loading letter by letter.
    url = ids(state)
    __w = requests.get(url).text
    _aw = json.loads(__w)["ids"]
    raw = _aw[:sample] if sample else _aw

    # Create `Submission` objects for each of the retrieved objects.
    # Getting the individual plans is the bottleneck here, and
    # unfortunately we can't retrieve them in bulk (... or can we?).
    submissions = []
    for entity in raw:
        # Retrieve the required data points.
        identifier = parse_id(entity["link"], df=False)
        districtr = individual(identifier)

        # Force all plan keys and values to strings.
        try:
            plan = {
                str(k): str(v) if not isinstance(v, list) else str(v[0])
                for k, v in districtr["plan"]["assignment"].items()
            }
            units = districtr["plan"]["units"]["name"]
            unitsType = districtr["plan"]["units"]["unitType"]
            tileset = districtr["plan"]["units"]["tilesets"][0]["sourceLayer"]

            # Create a new Submission.
            submissions.append(
                Submission(
                    link=entity["link"],
                    id=identifier,
                    plan=plan,
                    units=units,
                    unitsType=unitsType,
                    tileset=tileset,
                    type=entity["type"],
                )
            )
        except BaseException:
            pass

    return submissions


def as_dataframe(url) -> pd.DataFrame:
    """
    Retrieves encoded submission data from the provided URL and parses it into
    a pandas `DataFrame`.

    Args:
        url (str): Wherever we're getting things from.
    """
    raw = requests.get(url).content
    return pd.read_csv(io.StringIO(raw.decode("utf-8")), parse_dates=True)


def individual(identifier) -> dict:
    """
    Retrieves districtr data for an individual plan.

    Args:
        identifier (str): districtr identifier for an individual plan.

    Returns:
        districtr plan object (as a dictionary).
    """
    raw = requests.get(one(identifier))
    return json.loads(raw.text)


def parse_id(link, df=True) -> Union[str, pd.Series]:
    """
    Given a districtr link, parse out the districtr identifier.

    Args:
        l (str): districtr url containing the districtr ID of the provided
            plan.
        df (bool, optional): If `l` is a dataframe, then we use pandas string
            operations rather than built-in ones.

    Returns:
        districtr ID.
    """
    if df:
        return link.str.split("/").str[-1].str.split("?").str[0]
    return link.split("/")[-1].split("?")[0]


def parse_datetime(d) -> pd.Series:
    """
    Parses the timestamps in the dataframe returned by `as_dataframe()`.

    Args:
        d (str): Column of the dataframe containing timestamps.

    Returns:
        `d` with its datetimes parsed correctly.
    """
    # Parse datetimes.
    prefix = d.str.split("+").str[0]
    suffix = d.str.split("+").str[1].str.split(" ").str[0]
    dt = prefix + " +" + suffix

    # Convert datetimes.
    return dt.apply(lambda r: datetime.strptime(r, "%a %b %d %Y %X %Z %z"))
