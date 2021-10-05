
import geopandas as gpd
from evaltools import dualgraph

vtds = gpd.read_file("./test-resources/test-vtds")
vtds = vtds[[
    "COUNTYFP20", "GEOID20", "NAME20", "TOTPOP", "INTPTLAT20", "INTPTLON20",
    "G20PREDBID", "CONGRESS", "geometry"
]]
vtds["INTPTLAT20"] = vtds["INTPTLAT20"].astype(float)
vtds["INTPTLON20"] = vtds["INTPTLON20"].astype(float)

dg = dualgraph(vtds, "GEOID20")
dg.to_json("./test-resources/test-graph.json")
