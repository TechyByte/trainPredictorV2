print("import configuration")
import config

print("import network model, deriving the graph and geospatial data")
import network_model
# yields "bare_network_model.pkl"

print("import schedule, gathering schedule data (assumed SQL configured correctly")
import schedule

print("import incidents data")
import incidents

print("import compile_historical, gathering historical weather data and saving a populated historical network model to disk")
import compile_historical
# yields "historical_model.pkl"

print("import learn, training a model on historical data")
#import learn
print("skipping due to long runtime, uncomment to run")
# yields "trained_model_whole.h5"

print("import predict, making predictions on the model using services")
import predict

from datetime import date

#services = schedule.get_all_scheduled_services_on_date(date.today())
services = schedule.get_scheduled_services_on_date_through_tiploc(date.today(), "EXETRSD")
services = predict.make_predictions_on_model_using_services(services)
#yields a list of ScheduledService objects
#also saves predictions to file in "predicted_services.pkl" and "predicted_network_model.pkl"

relevant_sub_network_model = network_model.G.subgraph(schedule.get_relevant_tiplocs_from_services(services))

#example of how to get the latlong of a node
print(relevant_sub_network_model.nodes(data=True)["EXETRSD"]["latlong"])

#example of how to get the movements of a service
print(services[0].get_movements())  # list of Movement objects

#example of how to get the delay of a movement
print(services[0].get_movements()[0].predicted_delay)  # float  # None if no prediction

#example of how to get the tiploc of a movement
print(services[0].get_movements()[0].tiploc)  # string

#example of how to get the time of a movement
print(services[0].get_movements()[0].time)  # datetime.time

#example of how to get the train service code of a service
print(services[0].tsc)  # string

#example of how to get the atoc code of a service
print(services[0].atoc_code)  # string

#example of how to get the train identity of a service
print(services[0].train_identity)  # string

#example of how to get the uid of a service
print(services[0].uid)  # string

#example of how to get the origin tiploc of a service
print(services[0].get_origin().tiploc)  # string

#example of how to get the destination tiploc of a service
print(services[0].get_destination().tiploc)  # string
