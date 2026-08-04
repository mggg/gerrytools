"""`_ArtistRegistry` — tracks gerrytools-managed matplotlib artists.

Owned by `GerryPlotBase` and `GeoPlotBase` so rebuild flows can remove only the
artists gerrytools created on its axes, leaving any external artists (user
overlays, neighbouring subplot content, etc.) untouched.

External artists are never added to the registry. `remove_all()` therefore
never touches them; that is the load-bearing contract that replaces
`ax.clear()`.
"""

from __future__ import annotations

from collections.abc import Iterable

from matplotlib.artist import Artist
from matplotlib.container import Container


class _ArtistRegistry:
    """Track and remove gerrytools-managed artists on an axes."""

    def __init__(self) -> None:
        self._tracked: list[Artist | Container] = []

    def track(self, artist: Artist | Container | Iterable[Artist] | None) -> None:
        """Record one or more artists as gerrytools-managed.

        Accepts ``None`` for ergonomics: callers can pass the return value of a
        matplotlib API that may or may not have produced an artist without
        guarding the call site.
        """
        if artist is None:
            return
        if isinstance(artist, (Artist, Container)):
            self._tracked.append(artist)
            return
        for item in artist:
            if item is None:
                continue
            self._tracked.append(item)

    def remove_all(self) -> None:
        """Remove every tracked artist from its axes.

        Removal exceptions are swallowed per-artist because some matplotlib
        container artists (e.g. transient tick artists, already-detached
        artists) raise ``NotImplementedError`` or ``ValueError`` on
        ``.remove()`` even though the call is correct.
        """
        for artist in self._tracked:
            try:
                artist.remove()
            except (NotImplementedError, ValueError):
                pass
        self._tracked.clear()
