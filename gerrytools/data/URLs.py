import us


def ids(state):
    """
    URL for accessing districtr identifiers.

    Args:
        state: Name of the state (e.g. `"wisconsin"`) for which we're retrieving
            districtr identifiers.

    Returns:
        String with the appropriate URL.
    """
    # If we're in michigan, then we use the 'beta' pipeline instead of the 'prod'
    # one.
    pipeline = "beta" if state == us.states.MI else "prod"
    if state == us.states.MI:
        prefix = "https://o1siz7rw0c.execute-api.us-east-2.amazonaws.com"
    else:
        prefix = "https://k61e3cz2ni.execute-api.us-east-2.amazonaws.com"

    return f"{prefix}/{pipeline}/submissions/districtr-ids/{state.name.lower()}"


def csvs(state, ptype="plan"):
    """
    URL for accessing districtr plan metadata.

    Args:
        state: `us.States` object (e.g. `us.states.WI`)
        ptype: Type of plan we're retrieving; defaults to `"plan"`.

    Returns:
        String with the appropriate URL.
    """
    pipeline = "beta" if state == us.states.MI else "prod"
    if state == us.states.MI:
        prefix = "https://o1siz7rw0c.execute-api.us-east-2.amazonaws.com"
    else:
        prefix = "https://k61e3cz2ni.execute-api.us-east-2.amazonaws.com"

    suffix = f"?type={ptype}&length=10000"

    return f"{prefix}/{pipeline}/submissions/csv/{state.name.lower()}{suffix}"


def one(identifier):
    """
    URL for accessing an individual districtr plan.

    Args:
        identifier: districtr identifier.

    Returns:
        String with the appropriate URL.
    """
    return f"https://districtr.org/.netlify/functions/planRead?id={identifier}"
