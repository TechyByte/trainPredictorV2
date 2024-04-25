import os
import time
from datetime import datetime
import logging
import config
import pandas as pd

CANCELLED_EVENT_CODES = ["C", "D", "O", "P", "S", "F"]

COLUMNS_TO_REMOVE = ["FINANCIAL_YEAR_PERIOD",
                     "PLANNED_ORIGIN_GBTT_DATETIME",
                     "PLANNED_DEST_GBTT_DATETIME",
                     "TRAILING_LOAD",
                     "TIMING_LOAD",
                     "UNIT_CLASS",
                     "INCIDENT_CREATE_DATE",
                     "NR_LOCATION_MANAGER",
                     "RESPONSIBLE_MANAGER",
                     "ATTRIBUTION_STATUS",
                     "INCIDENT_EQUIPMENT",
                     "INCIDENT_DESCRIPTION",
                     "REACT_REASON"]

TRANSPARENCY_REPORTS_DIRECTORY = "input_files/transparency_reports/"

df_list = []

datetime_columns = ["PLANNED_ORIGIN_WTT_DATETIME", "PLANNED_DEST_WTT_DATETIME",
                    "INCIDENT_START_DATETIME", "INCIDENT_END_DATETIME", "EVENT_DATETIME"]

logging.info("Reading transparency reports...")
tic = time.perf_counter()


def get_earliest_incident():
    return df["PLANNED_ORIGIN_WTT_DATETIME"].min()


def get_latest_incident():
    return df["PLANNED_DEST_WTT_DATETIME"].max()



def dateparse(x):
    try:
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return datetime.strptime(x, '%d-%b-%Y %H:%Ms')


for file in os.listdir(TRANSPARENCY_REPORTS_DIRECTORY):
    if file.endswith(".csv"):
        df_list.append(pd.read_csv(TRANSPARENCY_REPORTS_DIRECTORY + file, low_memory=False,
                                   # dtype={column: "datetime64[ns]" for column in datetime_columns},
                                   # parse_dates=datetime_columns,
                                   # date_parser=dateparse
                                   ))

df = pd.concat(df_list, axis=0, ignore_index=True)

for column in datetime_columns:
    df[column] = pd.to_datetime(df[column], format="%Y-%m-%d %H:%M:%S", errors="ignore")
    df[column] = pd.to_datetime(df[column], format="%d-%b-%Y %H:%M", errors="ignore")

# Replace [NON_]PFPI_MINUTES with max value where EVENT_TYPE indicates a cancellation, diversion or otherwise failed service
df.loc[(df["EVENT_TYPE"].isin(CANCELLED_EVENT_CODES)), "PFPI_MINUTES"] = max(df["PFPI_MINUTES"])
df.loc[(df["EVENT_TYPE"].isin(CANCELLED_EVENT_CODES)), "NON_PFPI_MINUTES"] = max(df["NON_PFPI_MINUTES"])

toc = time.perf_counter()
logging.info(f"Processing transparency reports... {toc - tic:0.4f} seconds so far")

for column in COLUMNS_TO_REMOVE:
    df.drop(column, axis=1, inplace=True)

toc = time.perf_counter()
logging.info(f"Transparency reports ready (took {toc - tic:0.4f} seconds)")

if __name__ == "__main__":
    print(df)
    print(df.columns)
    print(df.dtypes)
    print(df.shape)
    print(get_earliest_incident())
    print(get_latest_incident())
