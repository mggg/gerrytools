import httpx
import pytest

import gerrytools.data.uscensus._api as census_api
from tests.data._helpers import MockHTTP


@pytest.fixture(autouse=True)
def recorded_retry_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry backoff sleeps instead of waiting, keeping retry-path tests instant."""

    recorded: list[float] = []
    monkeypatch.setattr(census_api, "_sleep", recorded.append)
    return recorded


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> MockHTTP:
    """Patch ``httpx.Client`` so all clients use a recording mock transport.

    Every ``httpx.Client(...)`` built while this fixture is active is given an
    ``httpx.MockTransport`` bound to the returned ``MockHTTP``, so no request escapes to the
    network and all are recorded for assertions.
    """

    mock = MockHTTP()
    real_client = httpx.Client

    def client_factory(*args, **kwargs) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(mock.handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    return mock
