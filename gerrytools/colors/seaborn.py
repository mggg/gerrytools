import seaborn as sns


def _seaborn_palette(
    name: str, n: int, *, reverse: bool = False
) -> list[tuple[float, float, float]]:
    """One validated core for the seaborn palette wrappers below."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    colors = list(sns.color_palette(name, n_colors=n))
    return list(reversed(colors)) if reverse else colors


def redbluecmap(n: int) -> list[tuple[float, float, float]]:
    """Generate a red/white/blue palette with ``n`` colors.

    Uses seaborn's reversed ``bwr`` diverging colormap so red comes first.

    Args:
        n (int): The number of colors to generate.

    Returns:
        list[tuple[float, float, float]]: List of RGB triples (each in [0, 1]).
    """
    return _seaborn_palette("bwr", n, reverse=True)


def greenpurplecmap(n: int) -> list[tuple[float, float, float]]:
    """Generate a green/white/purple palette with ``n`` colors.

    Uses seaborn's reversed ``PRGn`` diverging colormap so green comes first.

    Args:
        n (int): The number of colors to generate.

    Returns:
        list[tuple[float, float, float]]: List of RGB triples (each in [0, 1]).
    """
    colors = _seaborn_palette("PRGn", n, reverse=True)

    # Use a consistent light neutral midpoint.
    if n % 2 == 1:
        colors[n // 2] = (240 / 255, 240 / 255, 240 / 255)

    return colors


def flare(n: int) -> list[tuple[float, float, float]]:
    """Generate a red-to-purple palette with ``n`` colors using seaborn's ``flare`` colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        list[tuple[float, float, float]]: List of RGB triples (each in [0, 1]).
    """
    return _seaborn_palette("flare", n)


def purples(n: int) -> list[tuple[float, float, float]]:
    """Generate ``n`` purple shades using the Matplotlib/seaborn ``Purples`` colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        list[tuple[float, float, float]]: List of RGB triples (each in [0, 1]).
    """
    return _seaborn_palette("Purples", n)


def greens(n: int) -> list[tuple[float, float, float]]:
    """Generate ``n`` green shades using the Matplotlib/seaborn ``Greens`` colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        list[tuple[float, float, float]]: List of RGB triples (each in [0, 1]).
    """
    return _seaborn_palette("Greens", n)
