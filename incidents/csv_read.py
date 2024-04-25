import pandas as pd

columns_to_remove = ["FINANCIAL_YEAR_PERIOD",
                     "PLANNED_ORIGIN_GBTT_DATETIME",
                     "PLANNED_DEST_GBTT_DATETIME",
                     "TRAILING_LOAD",
                     "TIMING_LOAD",
                     "UNIT_CLASS",
                     "INCIDENT_CREATE_DATE",
                     "NR_LOCATION_MANAGER",
                     "RESPONSIBLE_MANAGER",
                     "ATTRIBUTION_STATUS",
                     "INCIDENT_DESCRIPTION",
                     "REACT_REASON",

]

df = pd.read_csv("Transparency-23-24-P01.csv")
print(df.head())