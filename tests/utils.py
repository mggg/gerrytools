
import requests
import tempfile
import json
from gerrychain import Graph

def remotegraphresource(file):
    r = requests.get(remoteresource(file)).json()
    temp = tempfile.NamedTemporaryFile(mode="w+")
    json.dump(r, temp)
    temp.flush()

    return Graph.from_json(temp.name)

def remoteresource(file):
    return f"https://evaltools-test-data.s3.amazonaws.com/{file}"