import datetime

import MySQLdb

import config
import network_model

from schedule.scheduled_service import ScheduledService

# TODO: Get scheduled services

# TODO: Filter scheduled services based on query parameters
# TODO: Build SQL query based on the query parameters
# Second tiploc is optional and means an edge should be queried instead of a node (bidirectional)

# TODO: Return the filtered scheduled services
db = MySQLdb.connect(host=config.db_host, user=config.db_user, passwd=config.db_password, db=config.db_database)
c = db.cursor()

def get_all_scheduled_services_on_date(date):
    services = []
    two_letter_day = tld(date)
    standard_initial_query = f"SELECT id, train_uid, train_identity, train_service_code, atoc_code FROM `schedule` WHERE `start_date` <= '{date}' AND `end_date` >= '{date}' AND `runs_{two_letter_day}` = 1;"
    c.execute(standard_initial_query)
    for row in c.fetchall():
        # for each  service
        service = ScheduledService(row[1], row[2], row[3], row[4])
        location_query = f"SELECT tiploc_code, arrival, pass, departure FROM `location` WHERE id = {row[0]} ORDER BY `order` ASC;"
        c.execute(location_query)
        for location in c.fetchall():
            if location[2] is None:
                if location[1] is None:
                    time = location[3]
                else:
                    time = location[1]
            else:
                time = location[2]
            service.add_stop(location[0], time)
        services.append(service)
    return services


def get_scheduled_services_on_date_through_tiploc(date, tiploc, origin=False, destination=False, train_service_code=None,
                                                  also_passing_tiploc=None) -> [ScheduledService]:
    services = []
    two_letter_day = tld(date)
    if origin:
        origin_supp = " AND WHERE `order` = 1"
    else:
        # TODO: fix below
        origin_supp = ""
    if destination:
        # TODO: below
        destination_supp = " AND WHERE `order` IN (SELECT `order` FROM `location` WHERE "
    else:
        destination_supp = ""
    standard_initial_query = f"SELECT id, train_uid, train_identity, train_service_code, atoc_code FROM `schedule` WHERE `start_date` <= '{date}' AND `end_date` >= '{date}' AND `runs_{two_letter_day}` = 1 AND `id` IN (SELECT id FROM `location` WHERE `tiploc_code` = '{tiploc}'{origin_supp}{destination_supp});"
    # returns all services passing through a tiploc on the given date

    # To make this query only an origin, also specify that orderID is 1 in query
    # To make this query only a destination, also specify that orderID of given TIPLOC is the max(orderID) for each scheduled service
    # To make this query an edge, specify that tiploc_1's orderID should be 1 more or less than tiploc_2's orderID
    # TODO: For also_passing_tiploc, iterate over result and discard all that don't include it

    c.execute(standard_initial_query)
    for row in c.fetchall():
        past_nodes = []
        # for each potentially matching service
        service = ScheduledService(row[1], row[2], row[3], row[4])
        location_query = f"SELECT tiploc_code, arrival, pass, departure FROM `location` WHERE id = {row[0]} ORDER BY `order` ASC;"
        c.execute(location_query)
        for location in c.fetchall():
            if location[2] is None:
                if location[1] is None:
                    time = location[3]
                else:
                    time = location[1]
            else:
                time = location[2]

            service.add_stop(location[0], time)
            past_nodes.append(location[0])
        if also_passing_tiploc is not None:
            if len(list(set(past_nodes).intersection([tiploc, also_passing_tiploc]))) >= 2:
                services.append(service)
        else:
            services.append(service)
    return services


def tld(date):
    return date.strftime("%A")[:2].lower()


if __name__ == "__main__":
    test = get_scheduled_services_on_date_through_tiploc(datetime.date.today(), "EXETRSD", also_passing_tiploc="CWLYBDG")
    print(test)
    print(f"Services found: {len(test)}")


def get_relevant_tiplocs_from_services(services):
    relevant_tiplocs = []
    for service in services:
        movements = service.get_movements()
        for movement in movements:
            if movement.tiploc not in relevant_tiplocs:
                relevant_tiplocs.append(movement.tiploc)
                try:
                    network_model.G.nodes(data=True)[movement.tiploc]['movements'] = []
                except KeyError:
                    pass
    return relevant_tiplocs
