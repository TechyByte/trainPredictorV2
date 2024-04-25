import networkx as nx
import json
import matplotlib.pyplot as plt
import scipy

import incidents.csv_read
from utilities import BiDict

G = nx.DiGraph()

tiploc_stanox = BiDict()


def load_toc_file(f):
    for line in f:
        data = json.loads(line)
        try:
            tiploc_entry = data["TiplocV1"]
            tiploc_stanox[tiploc_entry["tiploc_code"]] = tiploc_entry["stanox"]
        except:
            try:
                schedule_entry = data["JsonScheduleV1"]
                previous_stop = None
                for stop in schedule_entry["schedule_segment"]["schedule_location"]:
                    if previous_stop is None:
                        previous_stop = stop["tiploc_code"]
                    else:
                        G.add_edge(previous_stop, stop["tiploc_code"])
                        previous_stop = stop["tiploc_code"]

            except:
                continue
    for k, v in tiploc_stanox.items():
        try:
            G.nodes[k]["stanox"] = v
        except KeyError:
            continue


def get_routes(origin, destination):
    paths = []
    try:
        for path in nx.all_shortest_paths(G, origin, destination):
            paths.append(path)
            print(path)
        return paths
    except KeyError:
        print("Error")


print("Getting ready...")
load_toc_file(open("toc-full.json"))
print("Let's go!")


if __name__ == "__main__":
    get_routes("EXETRSD", "TIVIPW")
    get_routes("EXETRSD", "BHAMNWS")
    get_routes("EXETRSD", "EXMOUTH")
    #print((G.nodes["DIGBY"]).adjacents())
    print(G.nodes["EXETRSD"])
    print(G.nodes["EXETRSD"]["stanox"])
