from gerrychain import Graph
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt


for st in ["mi", "wi", "va"]:
    shapes = gpd.read_file("../data/shapes_no_water_vtds/{}_vtds_overlay_water.shp".format(st))
    shapes = shapes.set_index("GEOID20").rename(columns={"COUNTYFP20": "COUNTY"})
    shapes.geometry = shapes.buffer(0)
    graph = Graph.from_geodataframe(shapes)
    graph.to_json("../dual_graphs/{}_vtds.json".format(st))


mi_g = Graph.from_json("../dual_graphs/mi_vtds.json")
wi_g = Graph.from_json("../dual_graphs/wi_vtds.json")
va_g = Graph.from_json("../dual_graphs/va_vtds.json")

pos = lambda g: {n: (float(g.nodes()[n]["INTPTLON20"]), float(g.nodes()[n]["INTPTLAT20"])) for n in g.nodes()}

nx.draw(wi_g, pos=pos(wi_g), node_size=1)
plt.savefig("wi_dual_graph.png", dpi=200)
nx.draw(va_g, pos=pos(va_g), node_size=1)
plt.savefig("va_dual_graph.png", dpi=200)
nx.draw(mi_g, pos=pos(mi_g), node_size=1)

mi_components = [g for g in nx.connected_components(mi_g)]
for c in mi_components:
    g = mi_g.subgraph(c)
    plt.figure()
    nx.draw(g, pos=pos(mi_g), node_size=1)


## Mackinac Bridge
mi_g.add_edge("26097097010", "26031031014")

## Groose Island to mainland Detroit
mi_g.add_edge("26163163851", "26163163670")
mi_g.add_edge("26163163909", "26163163671")

## Beaver Island to Charelevoix
mi_g.add_edge("26029029018", "26029029006")

nx.draw(mi_g, pos=pos(mi_g), node_size=1)
plt.savefig("mi_dual_graph.png", dpi=200)
mi_g.to_json("../dual_graphs/mi_vtds.json")