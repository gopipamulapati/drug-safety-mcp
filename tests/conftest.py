import pytest

from drug_safety_mcp import server
from drug_safety_mcp.offline import offline_transport
from drug_safety_mcp.openfda import OpenFDAClient


@pytest.fixture
def offline_client():
    client = OpenFDAClient(api_key="", transport=offline_transport())
    server.set_client(client)
    yield client
    server.set_client(None)
