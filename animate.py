from datetime import date, datetime

import pandas as pd
import shapely
from matplotlib import pyplot as plt
from pandas import Timestamp
import geopandas as gpd

import predict
import schedule
import network_model as nm
import networkx as nx

services = schedule.get_scheduled_services_on_date_through_tiploc(date.today(), "EXETRSD")
#services = predict.make_predictions_on_model_using_services(services)

sub_network_model = nm.G.subgraph(schedule.get_relevant_tiplocs_from_services(services))

columns = ["datetime", "lat1", "long1", "lat2", "long2", "delay"]
rows = []

for s in services:
    movements = s.get_movements()
    for i in range(len(movements)-1):
        t1 = Timestamp(datetime.combine(date.today(), (datetime.min + movements[i].time).time())).tz_localize("Europe/London")
        t2 = Timestamp(datetime.combine(date.today(), (datetime.min + movements[i+1].time).time())).tz_localize("Europe/London")
        t = t1 + (t2 - t1)/2
        ll1 = sub_network_model.nodes(data=True)[movements[i].tiploc]["latlong"]
        ll2 = sub_network_model.nodes(data=True)[movements[i+1].tiploc]["latlong"]
        del1 = movements[i].predicted_delay
        del2 = movements[i+1].predicted_delay
        try:
            delay = (del1+del2)/2
        except TypeError:
            delay = 0
        rows.append({"datetime": t, "lat1": ll1[0], "long1": ll1[1], "lat2": ll2[0], "long2": ll2[1], "delay": delay})


df = pd.DataFrame(rows, columns=columns)


# Create GeoDataFrames
gdf = gpd.GeoDataFrame(df, geometry=df.apply(
        lambda r: shapely.geometry.LineString(
            [(r["long1"], r["lat1"]),
             (r["long2"], r["lat2"])]), axis=1)
)

# Set datetime column as index
gdf.set_index('datetime', inplace=True)

gdf.sort_index(inplace=True, ascending=True)

print(f"Maximum delay: {gdf['delay'].max()}")

def get_color(delay_in_minutes):
    if delay_in_minutes < -0.5: # Early
        return "blue"
    elif delay_in_minutes < 0.5: # On time
        return "green"
    elif delay_in_minutes < 5:
        return "yellow"
    else:
        return "red"

print(gdf.columns)
print(gdf.head())



gdf['color'] = gdf['delay'].apply(get_color)











exit()


def convert(latlong):
    # reflect latlong for plottable coordinates
    return latlong[1], latlong[0]


positions = nx.get_node_attributes(sub_network_model, "latlong")
converted_positions = {node: convert(latlong) for node, latlong in positions.items()}

ax = plt.gca()
ax.set_aspect('equal', adjustable='datalim')

# edgelist = [e for e in sub_network_model.edges if e not in nx.selfloop_edges(sub_network_model)]

nx.draw_networkx_nodes(sub_network_model, pos=converted_positions, node_size=1)

# nx.draw_networkx_edges(nm.G, pos=converted_positions, edgelist=edgelist)

# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()

plt.savefig("sub_map.png")

