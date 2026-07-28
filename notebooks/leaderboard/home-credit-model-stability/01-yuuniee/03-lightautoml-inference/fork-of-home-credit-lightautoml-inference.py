# %% [code]
# %% [code]
# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T19:57:03.798605Z","iopub.execute_input":"2024-04-23T19:57:03.799004Z","iopub.status.idle":"2024-04-23T19:57:16.708897Z","shell.execute_reply.started":"2024-04-23T19:57:03.798976Z","shell.execute_reply":"2024-04-23T19:57:16.707889Z"},"jupyter":{"outputs_hidden":false}}
# %%time 

# # Source - 
# !cp /kaggle/usr/lib/fork_of_home_credit_risk_lightgbm/fork_of_home_credit_risk_lightgbm.py fork_of_home_credit_risk_lightgbm.py
# !python fork_of_home_credit_risk_lightgbm.py
# !mv submission.csv submission_me.csv

# # _ = Utils.CleanMemory(); 
# %reset -f

# import gc
# gc.collect()

# %% [markdown] {"papermill":{"duration":0.006713,"end_time":"2024-02-10T04:58:28.609783","exception":false,"start_time":"2024-02-10T04:58:28.60307","status":"completed"},"tags":[]}
# # Dependencies

# %% [code] {"scrolled":true,"_kg_hide-output":true,"execution":{"iopub.status.busy":"2024-04-23T19:57:16.710602Z","iopub.execute_input":"2024-04-23T19:57:16.71089Z","iopub.status.idle":"2024-04-23T19:59:49.193122Z","shell.execute_reply.started":"2024-04-23T19:57:16.710866Z","shell.execute_reply":"2024-04-23T19:59:49.192116Z"},"jupyter":{"outputs_hidden":false}}

# !pip install --no-index -Uq --find-links=/kaggle/input/lightautoml-038-dependencies lightautoml==0.3.8
import os
os.system('pip install --no-index -Uq --find-links=/kaggle/input/lightautoml-038-dependencies lightautoml==0.3.8')

# %% [code] {"papermill":{"duration":6.303403,"end_time":"2024-02-10T04:58:34.920027","exception":false,"start_time":"2024-02-10T04:58:28.616624","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-04-23T19:59:49.19441Z","iopub.execute_input":"2024-04-23T19:59:49.194707Z","iopub.status.idle":"2024-04-23T19:59:53.941494Z","shell.execute_reply.started":"2024-04-23T19:59:49.194679Z","shell.execute_reply":"2024-04-23T19:59:53.940476Z"},"jupyter":{"outputs_hidden":false}}
import os
import gc
from glob import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import lightgbm as lgb
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T19:59:53.94298Z","iopub.execute_input":"2024-04-23T19:59:53.944047Z","iopub.status.idle":"2024-04-23T19:59:53.957119Z","shell.execute_reply.started":"2024-04-23T19:59:53.94401Z","shell.execute_reply":"2024-04-23T19:59:53.956242Z"},"jupyter":{"outputs_hidden":false}}
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T19:59:53.959995Z","iopub.execute_input":"2024-04-23T19:59:53.960398Z","iopub.status.idle":"2024-04-23T20:00:20.706118Z","shell.execute_reply.started":"2024-04-23T19:59:53.960373Z","shell.execute_reply":"2024-04-23T20:00:20.705249Z"},"jupyter":{"outputs_hidden":false}}
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.metrics import mean_squared_error

automl_path = "/kaggle/input/hc-lama-models/"

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:20.70718Z","iopub.execute_input":"2024-04-23T20:00:20.70782Z","iopub.status.idle":"2024-04-23T20:00:29.351383Z","shell.execute_reply.started":"2024-04-23T20:00:20.707791Z","shell.execute_reply":"2024-04-23T20:00:29.350515Z"},"jupyter":{"outputs_hidden":false}}
# automl_model = joblib.load(f'{automl_path}denselight_model.pkl')
# feature_names = automl_model.collect_used_feats()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:29.352506Z","iopub.execute_input":"2024-04-23T20:00:29.352781Z","iopub.status.idle":"2024-04-23T20:00:29.358619Z","shell.execute_reply.started":"2024-04-23T20:00:29.352745Z","shell.execute_reply":"2024-04-23T20:00:29.357798Z"},"jupyter":{"outputs_hidden":false}}
# print(automl_model.collect_model_stats())

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:29.359916Z","iopub.execute_input":"2024-04-23T20:00:29.360194Z","iopub.status.idle":"2024-04-23T20:00:29.379755Z","shell.execute_reply.started":"2024-04-23T20:00:29.360171Z","shell.execute_reply":"2024-04-23T20:00:29.378963Z"},"jupyter":{"outputs_hidden":false}}
train_cols, cat_cols, drop_cols = joblib.load(f"{automl_path}train_cat_columns_in_WM.pkl")
train_cols = [n for n in train_cols if n != "target"]
print("train_cols:\t", len(train_cols))
print("cat_cols:\t", len(cat_cols))
print("drop_cols:\t", len(drop_cols))

# %% [markdown] {"papermill":{"duration":0.006776,"end_time":"2024-02-10T04:58:34.934015","exception":false,"start_time":"2024-02-10T04:58:34.927239","status":"completed"},"tags":[]}
# # Data collection

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:29.382835Z","iopub.execute_input":"2024-04-23T20:00:29.383103Z","iopub.status.idle":"2024-04-23T20:00:29.404379Z","shell.execute_reply.started":"2024-04-23T20:00:29.383081Z","shell.execute_reply":"2024-04-23T20:00:29.403479Z"},"jupyter":{"outputs_hidden":false}}
class Pipeline:
    @staticmethod
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))            

        return df
    
    @staticmethod
    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
                df = df.with_columns(pl.col(col).cast(pl.Float32))
                
        df = df.drop("date_decision", "MONTH")

        return df
    
#     @staticmethod
#     def filter_cols(df):
#         for col in df.columns:
#             if col not in ["target", "case_id", "WEEK_NUM"]:
#                 isnull = df[col].is_null().mean()

#                 # TODO: Revisar el sentido de este filtro         
#                 if col[-1]=='M':
#                     specific_value_ratio = df.filter(pl.col(col) == "a55475b1").height / df.height
#                     if specific_value_ratio > 0.95:
#                         df = df.drop(col)
                        
#                 if isnull > 0.95:
#                     df = df.drop(col)
                    
#         for col in df.columns:
#             if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
#                 freq = df[col].n_unique()

#                 if (freq == 1) | (freq > 200):
#                     df = df.drop(col)
            
#             # eliminate yaer, month feature
#             if (col[-1] not in ["P", "A", "L", "M"]) and (('month_' in col) or ('year_' in col)):
#                 df = df.drop(col)
        
#         # for col in cols_drop:
#         #     if col in df.columns:
#         #         df = df.drop(col)
            
#         return df

    
    # Añadidos los 3 siguientes metodos
    @staticmethod
    def reduce_memory_usage_pl(df):
        """ Reduce memory usage by polars dataframe {df} with name {name} by changing its data types.
            Original pandas version of this function: https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 
        """
        print(f"Memory usage of dataframe is {round(df.estimated_size('mb'), 2)} MB")
        
        Numeric_Int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
        Numeric_Float_types = [pl.Float32, pl.Float64]    
        
        for col in df.columns:
            if col == 'case_id': 
                continue
            try:
                col_type = df[col].dtype
                
                if col_type == pl.Categorical:
                    continue
                    
                c_min = df[col].min()
                c_max = df[col].max()
                
                if col_type in Numeric_Int_types:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df = df.with_columns(df[col].cast(pl.Int8))
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df = df.with_columns(df[col].cast(pl.Int16))
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df = df.with_columns(df[col].cast(pl.Int32))
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df = df.with_columns(df[col].cast(pl.Int64))
                
                elif col_type in Numeric_Float_types:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df = df.with_columns(df[col].cast(pl.Float32))
                    else:
                        pass
                # elif col_type == pl.Utf8:
                #     df = df.with_columns(df[col].cast(pl.Categorical))
                else:
                    pass
            except:
                pass
        print(f"Memory usage of dataframe became {round(df.estimated_size('mb'), 2)} MB")
        return df
    
    
    @staticmethod
    def fill_missing_values(df):
        num_cnt = 0
        cat_cnt = 0
        for col in df.columns:
            if df[col].dtype.is_numeric():
                df = df.with_columns(pl.col(col).fill_null(0).alias(col))
                num_cnt += 1
            else:
                df = df.with_columns(pl.col(col).fill_null("Missing").alias(col))
                cat_cnt += 1
        print("num_cnt : ", num_cnt)
        print("cat_cnt : ", cat_cnt)
        return df

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:29.405679Z","iopub.execute_input":"2024-04-23T20:00:29.405973Z","iopub.status.idle":"2024-04-23T20:00:29.421467Z","shell.execute_reply.started":"2024-04-23T20:00:29.40595Z","shell.execute_reply":"2024-04-23T20:00:29.420606Z"},"jupyter":{"outputs_hidden":false}}
class Aggregator:
    @staticmethod
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols]

        return expr_max + [pl.mean(col).alias(f"mean_{col}") for col in cols] + [pl.std(col).alias(f"std_{col}") for col in cols]

    @staticmethod
    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D",)] 
            
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols] # + [pl.sum(col).alias(f"sum_{col}") for col in cols]

        return expr_max # + [pl.mean(col).alias(f"mean_{col}") for col in cols]

    @staticmethod
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        
        expr_max = [pl.last(col).alias(f"last_{col}") for col in cols] + \
            [pl.n_unique(col).alias(f"n_unique_{col}") for col in cols] + \
            [pl.first(col).alias(f"first_{col}") for col in cols]  # High Value

        return expr_max

    @staticmethod
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols] + [pl.sum(col).alias(f"sum_{col}") for col in cols]

        return expr_max # + [pl.mean(col).alias(f"mean_{col}") for col in cols] + [pl.std(col).alias(f"std_{col}") for col in cols]
    
    @staticmethod
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols] # + [pl.n_unique(col).alias(f"n_unique_{col}") for col in cols]

        return expr_max

    @staticmethod
    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)

        return exprs

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:29.422588Z","iopub.execute_input":"2024-04-23T20:00:29.423017Z","iopub.status.idle":"2024-04-23T20:00:29.438155Z","shell.execute_reply.started":"2024-04-23T20:00:29.422987Z","shell.execute_reply":"2024-04-23T20:00:29.43735Z"},"jupyter":{"outputs_hidden":false}}
def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    
    if depth in [1, 2]:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.get_exprs(df))
        
    df = df.pipe(Pipeline.reduce_memory_usage_pl)
    return df

def read_files(regex_path, depth=None):
    chunks = []
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        
        if depth in [1, 2]:
            df = df.sort("num_group1").group_by("case_id").agg(Aggregator.get_exprs(df))
        
        chunks.append(df)
        
    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    
    df = df.pipe(Pipeline.reduce_memory_usage_pl)
    
    return df

def feature_eng(df_base, depth_0, depth_1, depth_2, is_train=True):
    df_base = (
        df_base
        .with_columns(
            decision_month = pl.col("date_decision").dt.month(),
            decision_weekday = pl.col("date_decision").dt.weekday(),
        )
    )
        
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
        
    df_base = df_base.pipe(Pipeline.handle_dates)
    if is_train:
        df_base = df_base.pipe(Pipeline.filter_cols)
    df_base = df_base.pipe(Pipeline.fill_missing_values)
    
    return df_base

def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    
    return df_data, cat_cols

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:29.439382Z","iopub.execute_input":"2024-04-23T20:00:29.439665Z","iopub.status.idle":"2024-04-23T20:00:29.453301Z","shell.execute_reply.started":"2024-04-23T20:00:29.439642Z","shell.execute_reply":"2024-04-23T20:00:29.452492Z"},"jupyter":{"outputs_hidden":false}}
from pathlib import Path
from glob import glob

ROOT            = Path("/kaggle/input/home-credit-credit-risk-model-stability")
TRAIN_DIR       = ROOT / "parquet_files" / "train"
TEST_DIR        = ROOT / "parquet_files" / "test"

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:29.468241Z","iopub.execute_input":"2024-04-23T20:00:29.468571Z","iopub.status.idle":"2024-04-23T20:00:29.770102Z","shell.execute_reply.started":"2024-04-23T20:00:29.468546Z","shell.execute_reply":"2024-04-23T20:00:29.769185Z"},"jupyter":{"outputs_hidden":false}}
data_store = {
    "df_base": read_file(TEST_DIR / "test_base.parquet"),
    "depth_0": [
        read_file(TEST_DIR / "test_static_cb_0.parquet"),
        read_files(TEST_DIR / "test_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
        read_files(TEST_DIR / "test_credit_bureau_a_1_*.parquet", 1),
        read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
        read_file(TEST_DIR / "test_other_1.parquet", 1),
        read_file(TEST_DIR / "test_person_1.parquet", 1),
        read_file(TEST_DIR / "test_deposit_1.parquet", 1),
        read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
        read_files(TEST_DIR / "test_credit_bureau_a_2_*.parquet", 2),
        read_file(TEST_DIR / "test_applprev_2.parquet", 2),
        read_file(TEST_DIR / "test_person_2.parquet", 2)
    ]
}

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:29.771113Z","iopub.execute_input":"2024-04-23T20:00:29.771383Z","iopub.status.idle":"2024-04-23T20:00:30.552755Z","shell.execute_reply.started":"2024-04-23T20:00:29.771359Z","shell.execute_reply":"2024-04-23T20:00:30.551903Z"},"jupyter":{"outputs_hidden":false}}
df_test = feature_eng(**data_store, is_train=False)
print("test data shape:\t", df_test.shape)
del data_store
gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:30.554245Z","iopub.execute_input":"2024-04-23T20:00:30.55473Z","iopub.status.idle":"2024-04-23T20:00:31.295693Z","shell.execute_reply.started":"2024-04-23T20:00:30.554693Z","shell.execute_reply":"2024-04-23T20:00:31.294812Z"},"jupyter":{"outputs_hidden":false}}
# df_test = df_test.select([col for col in df_train.columns if col != "target"])
df_test = df_test.select(train_cols)
# print("train data shape:\t", df_train.shape)
print("test data shape:\t", df_test.shape)

df_test, cat_cols = to_pandas(df_test, cat_cols)
# df_test = reduce_mem_usage(df_test)

gc.collect()

# %% [markdown] {"papermill":{"duration":0.031773,"end_time":"2024-02-10T05:23:03.16254","exception":false,"start_time":"2024-02-10T05:23:03.130767","status":"completed"},"tags":[]}
# # Prediction

# %% [code] {"papermill":{"duration":0.041537,"end_time":"2024-02-10T05:23:03.235674","exception":false,"start_time":"2024-02-10T05:23:03.194137","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-04-23T20:00:31.296794Z","iopub.execute_input":"2024-04-23T20:00:31.297057Z","iopub.status.idle":"2024-04-23T20:00:31.304282Z","shell.execute_reply.started":"2024-04-23T20:00:31.297034Z","shell.execute_reply":"2024-04-23T20:00:31.302994Z"},"jupyter":{"outputs_hidden":false}}
# def predict_proba_in_batches(data, batch_size=200000): # about 35min per 10k
#     num_samples = len(data)
#     num_batches = int(np.ceil(num_samples / batch_size))
#     probabilities = np.zeros((num_samples,))

#     for batch_idx in range(num_batches):
#         print(f"Processing batch: {batch_idx+1}/{num_batches}")
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, num_samples)
        
#         X_batch = data.iloc[start_idx:end_idx]
        
#         batch_probs_1 = joblib.load(f'{automl_path}denselight_model_in_WM_SEED3232.pkl').predict(X_batch).data.squeeze()
#         batch_probs_2 = joblib.load(f'{automl_path}denselight_model_in_WM.pkl').predict(X_batch).data.squeeze()
        
#         probabilities[start_idx:end_idx] = (batch_probs_2 + batch_probs_1)/2
        
#         del X_batch
#         gc.collect()

#     return probabilities
def predict_proba_in_batches(model, data, batch_size=5):
    num_samples = len(data)
    num_batches = int(np.ceil(num_samples / batch_size))
    probabilities = np.zeros((num_samples,))

    for batch_idx in range(num_batches):
        print(f"Processing batch: {batch_idx+1}/{num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        X_batch = data.iloc[start_idx:end_idx]
        batch_probs = model.predict(X_batch).data.squeeze()
        probabilities[start_idx:end_idx] = batch_probs
        gc.collect()

    return probabilities

# %% [code] {"papermill":{"duration":0.706127,"end_time":"2024-02-10T05:23:03.973644","exception":false,"start_time":"2024-02-10T05:23:03.267517","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-04-23T20:00:31.305628Z","iopub.execute_input":"2024-04-23T20:00:31.306416Z","iopub.status.idle":"2024-04-23T20:00:38.279658Z","shell.execute_reply.started":"2024-04-23T20:00:31.30639Z","shell.execute_reply":"2024-04-23T20:00:38.278569Z"},"jupyter":{"outputs_hidden":false}}
X_test = df_test.drop(columns=["WEEK_NUM"])
X_test = X_test.set_index("case_id")
print("X_test shape: ", df_test.shape)

automl_model = joblib.load(f'/kaggle/input/hc-lama-models/denselight_model_in_WM.pkl')
y_pred = pd.Series(predict_proba_in_batches(automl_model, X_test), index=X_test.index)
y_pred[:10]

# %% [markdown] {"papermill":{"duration":0.032096,"end_time":"2024-02-10T05:23:04.038858","exception":false,"start_time":"2024-02-10T05:23:04.006762","status":"completed"},"tags":[]}
# # Submission

# %% [code] {"papermill":{"duration":0.053299,"end_time":"2024-02-10T05:23:04.124231","exception":false,"start_time":"2024-02-10T05:23:04.070932","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-04-23T20:00:38.281051Z","iopub.execute_input":"2024-04-23T20:00:38.281457Z","iopub.status.idle":"2024-04-23T20:00:38.293178Z","shell.execute_reply.started":"2024-04-23T20:00:38.281427Z","shell.execute_reply":"2024-04-23T20:00:38.292356Z"},"jupyter":{"outputs_hidden":false}}
subm_df = pd.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/sample_submission.csv")
subm_df = subm_df.set_index("case_id")
subm_df["score"] = y_pred

# %% [code] {"execution":{"iopub.status.busy":"2024-04-23T20:00:38.314879Z","iopub.execute_input":"2024-04-23T20:00:38.31515Z","iopub.status.idle":"2024-04-23T20:00:38.32537Z","shell.execute_reply.started":"2024-04-23T20:00:38.315126Z","shell.execute_reply":"2024-04-23T20:00:38.324345Z"},"jupyter":{"outputs_hidden":false}}
print(subm_df)

# %% [code] {"papermill":{"duration":0.043978,"end_time":"2024-02-10T05:23:04.283155","exception":false,"start_time":"2024-02-10T05:23:04.239177","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-04-23T20:00:38.32668Z","iopub.execute_input":"2024-04-23T20:00:38.327036Z","iopub.status.idle":"2024-04-23T20:00:38.335942Z","shell.execute_reply.started":"2024-04-23T20:00:38.326989Z","shell.execute_reply":"2024-04-23T20:00:38.335128Z"},"jupyter":{"outputs_hidden":false}}
subm_df.to_csv("submission.csv")