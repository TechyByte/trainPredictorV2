from openmeteo_requests.Client import OpenMeteoRequestsError
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from time import sleep
from datetime import date, datetime
from sklearn.preprocessing import LabelEncoder
import schedule
from weather.openmeteo_collate import current_and_forecast
import logging

import weather

import network_model
import pickle

def make_predictions_on_model_using_services(services=schedule.get_all_scheduled_services_on_date(date.today()), model_file_name: str = 'trained_model_whole.h5'):
    # Load the saved model
    model = load_model(model_file_name)

    input_data = pd.DataFrame(columns=["tiploc", "time", "train_service_code", "weather"])
    count = 0
    requested_predictions = 0
    relevant_tiplocs = schedule.get_relevant_tiplocs_from_services(services)

    logging.info(f"Querying weather at {len(relevant_tiplocs)} locations")

    # Iterate over the nodes in the graph to add weather
    weather_count = 0
    retry_count = 3
    for node, data in network_model.G.nodes(data=True):
        # Check if the node is relevant and has a 'latlong' and 'weather_forecast" attributet
        if 'latlong' in data and node in relevant_tiplocs:
            weather_count += 1
            try:
                # Check if the node has 'current_weather' and 'weather_forecast' attributes
                test = data['weather_forecast']
            except KeyError:
                # Get the current weather and weather forecast at the node's location
                for i in range(retry_count):
                    try:
                        current_weather, weather_forecast = current_and_forecast(data['latlong'])
                        # Add the 'current_weather' and 'weather_forecast' attributes to the node
                        network_model.G.nodes[node]['current_weather'] = current_weather
                        network_model.G.nodes[node]['weather_forecast'] = weather_forecast.set_index('date')
                        break
                    except OpenMeteoRequestsError:
                        logging.error(f"Failed to retrieve weather data for {node} attempt {i + 1} of {retry_count}")
                        if i < (retry_count-1):
                            sleep((i+1)*60)

        if weather_count % int(len(relevant_tiplocs) / 100) == 0:
            logging.info(f"Processed {weather_count} of {len(relevant_tiplocs)} locations")

    from pandas import Timestamp

    total_movements = 0
    queried_services = []
    failed_weather_locations = []

    for service in services:
        count += 1
        movements = service.get_movements()
        total_movements += len(movements)
        queried_services.append(service)
        last_weather = None
        for movement in movements:
            train_service_code = service.tsc
            # Convert datetime.combine(date.today(), (datetime.min + movement.time).time()) to a Timestamp object
            timestamp = Timestamp(datetime.combine(date.today(), (datetime.min + movement.time).time())).tz_localize(
                "Europe/London")

            if "weather_forecast" in network_model.G.nodes(data=True)[movement.tiploc]:
                i = np.argmin(
                    np.abs(network_model.G.nodes(data=True)[movement.tiploc]['weather_forecast'].index - timestamp))

                closest_weather = network_model.G.nodes(data=True)[movement.tiploc]['weather_forecast'].iloc[i].fillna(0)
                current_weather = network_model.G.nodes(data=True)[movement.tiploc]['current_weather']
                new_row = pd.DataFrame({"tiploc": [movement.tiploc], "time": [date.today() + movement.time],
                                        "train_service_code": [service.tsc], "weather": [closest_weather.values.tolist()]})
                input_data = pd.concat([input_data, new_row], ignore_index=True)
                requested_predictions += 1
                last_weather = closest_weather
            else:
                if movement.tiploc not in failed_weather_locations:
                    failed_weather_locations.append(movement.tiploc)
                if last_weather is not None:
                    new_row = pd.DataFrame({"tiploc": [movement.tiploc], "time": [date.today() + movement.time],
                                            "train_service_code": [service.tsc], "weather": [last_weather.values.tolist()]})
                else:
                    new_row = pd.DataFrame({"tiploc": [movement.tiploc], "time": [date.today() + movement.time],
                                            "train_service_code": [service.tsc],
                                            "weather": [weather.__placeholder_weather()]})
                input_data = pd.concat([input_data, new_row], ignore_index=True)
                requested_predictions += 1
        if count > 5:
            pass
            # break # For testing purposes
        if count % 100 == 0:
            logging.info(f"Processed {count} services of {len(services)}")

    logging.info(f"Finished processing {count} services")

    # Initialize the LabelEncoder
    le = LabelEncoder()
    logging.info(f"Found {len(failed_weather_locations)} locations with missing weather data: {failed_weather_locations}")
    # Fit the LabelEncoder and transform the 'tiploc' and 'train_service_code' columns
    input_data["tiploc"] = le.fit_transform(input_data["tiploc"])
    # input_data["train_service_code"] = le.fit_transform(input_data["train_service_code"])


    # Convert the Unix timestamp to int64
    input_data['time'] = pd.to_datetime(input_data['time']).dt.tz_localize("Europe/London")
    for i in range(len(input_data)):
        input_data.loc[i, 'time'] = input_data.loc[i, 'time'].timestamp()

    # Convert 'time' column to numeric format
    input_data['time'] = pd.to_datetime(input_data['time']).dt.tz_localize("Europe/London")
    input_data['time'] = input_data['time'].apply(lambda x: x.timestamp()).astype('float32')

    # Replace None in weather with a default value
    default_weather = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Adjust this based on your needs

    # Convert the 'weather' column to a DataFrame
    weather_data = pd.DataFrame(input_data['weather'].to_list(),
                                columns=[f"weather_{i}" for i in range(len(input_data['weather'].iloc[0]))])

    # Concatenate weather_data with input_data
    input_data = pd.concat([input_data.drop('weather', axis=1), weather_data], axis=1)

    # Now, you can create the numpy array
    input_data_array = np.column_stack([input_data[col].values for col in input_data.columns]).astype('float32')

    # Reshape the data
    input_data_array = input_data_array.reshape(input_data_array.shape[0], 13, 1)

    # Normalize the features
    #scaler = MinMaxScaler()
    #input_data_array = scaler.fit_transform(input_data_array)

    logging.info("Making predictions")
    # Make a prediction
    prediction = model.predict(input_data_array)

    assert len(input_data["time"]) == len(prediction)
    logging.info(f"Needed predictions: {total_movements}")
    logging.info(f"Predictions received: {len(prediction)}")
    logging.info(f"Needed predictions skipped: {len(prediction) - total_movements}")
    logging.info(
        f"Missing predictions: {requested_predictions - len(prediction)} ({100 * (len(prediction) / requested_predictions):0.2f}%)")

    logging.info("Processing predictions")
    i = 0
    if len(prediction) == total_movements:
        for s in queried_services:
            for movement in s.get_movements():
                movement.add_predicted_delay(prediction[i])
                try:
                    network_model.G.nodes(data=True)[movement.tiploc]['movements'].append(movement.predicted_delay)
                except KeyError:
                    pass
                i += 1
        logging.info("Predictions processed")
    else:
        logging.error("Can't predict delays for all movements")

    assert queried_services[0].get_movements()[0].predicted_delay is not None

    with open('predicted_services.pkl', 'wb') as f:
        pickle.dump(queried_services, f)

    with open('predicted_network_model.pkl', 'wb') as f:
        pickle.dump(network_model.G, f)

    return queried_services


if __name__ == "__main__":
    #services = schedule.get_scheduled_services_on_date_through_tiploc(date.today(), "EXETRSD")
    services = schedule.get_all_scheduled_services_on_date(date.today())
    predicted_services = make_predictions_on_model_using_services(services)