import logging
import pickle
import time

import networkx.exception
import pandas as pd
from meteostat import Hourly
from meteostat import Stations as WeatherStations  # avoiding ambiguity with [train] stations

import incidents
import network_model

overall_tic = time.perf_counter()


def get_valid_tiploc_from_stanox(stanox, stanox2=None):
    try:
        tiplocs = network_model.tiploc_stanox.inverse[stanox]
        for tiploc in tiplocs:
            if network_model.G.has_node(tiploc):
                return tiploc
    except KeyError:
        logging.debug("Stanox " + stanox + " not found in network model.")
        try:
            if stanox2 is not None:
                for tiploc in network_model.tiploc_stanox.inverse[stanox2]:
                    if network_model.G.has_node(tiploc):
                        return tiploc
        except KeyError:
            logging.debug("Stanox " + stanox2 + " not found in network model.")
            return None
    return None


count = 0
total = len(incidents.df)
success = 0
tic = time.perf_counter()

for incident in incidents.df.itertuples():
    count += 1
    if count % int(total / 100) == 0:
        logging.info(f"Processing incidents: {100 * count / total:0.4f}% ({count} of {total})")
    incident_start_location = get_valid_tiploc_from_stanox(str(incident.START_STANOX).zfill(5),
                                                           str(incident.PLANNED_ORIGIN_LOCATION_CODE).zfill(5))
    incident_end_location = get_valid_tiploc_from_stanox(str(incident.END_STANOX).zfill(5),
                                                         str(incident.PLANNED_DEST_LOCATION_CODE).zfill(5))

    if incident_start_location is not None and incident_end_location is not None:
        try:
            paths = network_model.get_routes(incident_start_location, incident_end_location)
            for path in paths:
                for node in path:
                    network_model.G.nodes[node]["incidents"].append(incident)
                break # Only add the incident to the first path found
            logging.debug("Successfully processed incident between " + incident_start_location
                          + " and " + incident_end_location)
            success += 1
        except networkx.exception.NodeNotFound:
            logging.error("One of the stations in the incident was not found in the network model.")
        except networkx.exception.NetworkXNoPath:
            logging.error(f"No path found between {incident_start_location} and {incident_end_location}")
    else:
        logging.debug(
            "It was not possible to add incident with TSC " + str(incident.TRAIN_SERVICE_CODE) + " to the network model")
        logging.debug("Incident start location: " + str(incident_start_location))
        logging.debug("Incident end location: " + str(incident_end_location))

toc = time.perf_counter()
logging.info(
    f"{success} incidents processed out of {total} incidents ({100 * success / total:0.4f}%) in {toc - tic:0.4f} seconds")


weather_stations = WeatherStations()

for node in network_model.G.nodes:
    if "latlong" in network_model.G.nodes[node] and len(network_model.G.nodes[node]["incidents"]) > 0:
        relevant_weather_station = weather_stations.nearby(network_model.G.nodes[node]["latlong"][0],
                                                           network_model.G.nodes[node]["latlong"][1]).fetch(1)
        logging.debug("Obtaining weather history for " + network_model.G.nodes[node]["name"] + " from "
                     + relevant_weather_station["name"])
        network_model.G.nodes[node]["weather_history"] = Hourly(relevant_weather_station,
                                                                start=incidents.get_earliest_incident(),
                                                                end=incidents.get_latest_incident()).fetch().drop("tsun", axis=1)

logging.info("Weather data gathered")

logging.info("Collating incidents within network nodes")
tic = time.perf_counter()
for node in network_model.G.nodes():
    # Convert existing list of incidents to a single DataFrame
    network_model.G.nodes[node]["incidents"] = pd.DataFrame(network_model.G.nodes[node]["incidents"])

toc = time.perf_counter()
logging.info(f"Collated incidents in {toc - tic:0.4f} seconds")

logging.info("Saving populated network model to file")

filename = "huge_raw_model.pkl"

with open(filename, "wb") as file:
    pickle.dump(network_model.G, file)

logging.info(f"Network model successfully saved to file: {filename}")

overall_toc = time.perf_counter()

logging.info(f"Total compilation time: {(overall_toc - overall_tic)/60:0.2f} minutes")