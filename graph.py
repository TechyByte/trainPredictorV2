import logging

import networkx
from matplotlib import pyplot as plt

import config

import network_model as nm



def convert(latlong):
    # reflect latlong for plottable coordinates
    return latlong[1], latlong[0]


positions = networkx.get_node_attributes(nm.G, "latlong")
converted_positions = {node: convert(latlong) for node, latlong in positions.items()}

ax = plt.gca()
ax.set_aspect('equal', adjustable='datalim')

edgelist = [e for e in nm.G.edges if e not in networkx.selfloop_edges(nm.G)]

networkx.draw_networkx_nodes(nm.G, pos=converted_positions, node_size=1)

# networkx.draw_networkx_edges(nm.G, pos=converted_positions, edgelist=edgelist)

# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()

plt.savefig("map.png")
logging.info("saved map.png")
from keras.models import load_model

# trained_model = load_model('trained_model.h5')
