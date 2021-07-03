import requests
import json
import pandas as pd
import numpy as np

from typing import List, Dict, Tuple

def arr_to_df(arr: List) -> pd.DataFrame:
    columns = arr[0][1:]
    data = [i[1:] for i in arr[1:]]
    index = [i[0] for i in arr[1:]]

    return pd.DataFrame(data, columns=columns, index=index).replace('', np.nan, inplace=True)

def df_to_arr(df: pd.DataFrame) -> List:
    data_df = df.replace(np.nan, '')

    data_df_index = data_df.index.tolist()
    data_list = data_df.to_numpy().tolist()

    data_list_with_index = [[data_df_index[i]] + row for i, row in enumerate(data_list)]
    data_list = [[''] + data_df.columns.tolist()] + data_list_with_index

    return data_list
def csv_to_df(file: str):
    data_df = pd.read_csv(file, index_col=0)
    return df_to_arr(data_df)

def arr_to_csv(file: str, arr: List):
    data_df = arr_to_df(arr)
    data_df.to_csv(file)

data_preferences = csv_to_df("in_preferences.csv")
data_activities = csv_to_df("in_activities.csv")
data_slots = csv_to_df("in_slots.csv")

url = "http://localhost:8000"
json_data = {
          "preferences": data_preferences,
          "slots": data_slots,
          "activities": data_activities
      }

response = json.loads(requests.post(url, json=json_data).content)

arr_to_csv("out2_plan_activities.csv", response["plan_activities"])
arr_to_csv("out2_plan_persons.csv", response["plan_persons"])

print(response)