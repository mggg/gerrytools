def sort_elections(elec_list):
    """
    Helper function to sort elections chronologically for plotting. Assumes the last two characters
    in the election name are the year, e.g. "SEN18"
    """
    tuplified_elecs = list(map(lambda x: (x[:-2], x[-2:]), sorted(elec_list)))
    sorted_tuples = sorted(tuplified_elecs, key=lambda x: x[1])
    return [tup[0] + tup[1] for tup in sorted_tuples]
