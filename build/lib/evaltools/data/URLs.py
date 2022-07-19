
def ids(state):
    """
    URL for accessing districtr identifiers.

    Args:
        state: Name of the state (e.g. `"wisconsin"`) for which we're retrieving
            districtr identifiers.

    Returns:
        String with the appropriate URL.
    """
    return f"https://k61e3cz2ni.execute-api.us-east-2.amazonaws.com/prod/submissions/districtr-ids/{state.name.lower()}"


def csvs(state, ptype="plan"):
    """
    URL for accessing districtr plan metadata.

    Args:
        state: `us.States` object (e.g. `us.states.WI`)
        ptype: Type of plan we're retrieving; defaults to `"plan"`.

    Returns:
        String with the appropriate URL.
    """
    prefix = "https://k61e3cz2ni.execute-api.us-east-2.amazonaws.com/prod/submissions/csv/"
    suffix = f"?type={ptype}&length=10000"
    return f"{prefix}{state.name.lower()}{suffix}"


def one(identifier):
    """
    URL for accessing an individual districtr plan.

    Args:
        identifier: distrivtr identifier.

    Returns:
        String with the appropriate URL.
    """
    return f"https://districtr.org/.netlify/functions/planRead?id={identifier}"

