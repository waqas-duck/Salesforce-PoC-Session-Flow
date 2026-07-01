#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General utility helpers for SessionFlow.
"""

import logging
import sys

import boto3
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO):
    """Configure terminal logging for scripts that use this utility module."""
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


#################################
def show_object_memory(obj, name=None, unit="auto", deep=True):
    """Print and return the approximate memory used by an object."""
    def get_recursive_size(value, seen):
        obj_id = id(value)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(value)
        if isinstance(value, dict):
            size += sum(
                get_recursive_size(key, seen) + get_recursive_size(val, seen)
                for key, val in value.items()
            )
        elif isinstance(value, (list, tuple, set, frozenset)):
            size += sum(get_recursive_size(item, seen) for item in value)
        elif hasattr(value, "__dict__"):
            size += get_recursive_size(vars(value), seen)
        elif hasattr(value, "__slots__"):
            slots = value.__slots__
            if isinstance(slots, str):
                slots = [slots]
            for slot in slots:
                if hasattr(value, slot):
                    size += get_recursive_size(getattr(value, slot), seen)

        return size

    def get_object_size(value):
        if isinstance(value, pd.DataFrame):
            return int(value.memory_usage(index=True, deep=deep).sum())
        if isinstance(value, pd.Series):
            return int(value.memory_usage(index=True, deep=deep))
        if isinstance(value, pd.Index):
            return int(value.memory_usage(deep=deep))
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if all(hasattr(value, attr) for attr in ("data", "indices", "indptr")):
            return int(value.data.nbytes + value.indices.nbytes + value.indptr.nbytes)
        if all(hasattr(value, attr) for attr in ("data", "row", "col")):
            return int(value.data.nbytes + value.row.nbytes + value.col.nbytes)

        return get_recursive_size(value, set())

    def format_size(size_bytes):
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_name = unit.upper()
        if unit_name != "AUTO":
            if unit_name not in units:
                raise ValueError(f"unit must be one of {['auto'] + units}")
            return size_bytes / (1024 ** units.index(unit_name)), unit_name

        unit_index = 0
        size = float(size_bytes)
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        return size, units[unit_index]

    size_bytes = get_object_size(obj)
    size, unit_name = format_size(size_bytes)
    label = name if name is not None else type(obj).__name__
    if isinstance(obj, pd.DataFrame):
        label = f"{label} shape={obj.shape}"
    memory_text = f"{label}: {size:.2f} {unit_name} ({size_bytes:,} bytes)"

    return memory_text


#################################
def get_secret():
    secret_name = "mli-snowflake-keys"
    region_name = "us-east-2"

    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = get_secret_value_response['SecretString']
    return secret


#################################
def get_private_key_from_secrets_manager(
    secret_arn: str = "arn:aws:secretsmanager:us-east-2:211125482819:secret:mli-snowflake-private-key-lkKe7m",
    region_name: str = "us-east-2",
    remove_padding: bool = True,
    ) -> str:
    
    """Fetch the Snowflake private key PEM string from AWS Secrets Manager."""
    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_arn)
    secret = response["SecretString"]

    if remove_padding:
        lines = secret.splitlines()
        body_lines = [line for line in lines if not line.startswith("-----")]
        secret = " ".join(body_lines)

    return secret


#################################
def fetch_pandas_all(cur, sql):
    df = []
    cur.execute(sql)
    col_names = [col[0] for col in cur.description]
    rows = 0
    while True:
        dat = cur.fetchmany(50000)
        if not dat:
            break
        df_tmp = pd.DataFrame(dat, columns=col_names)
        rows += df_tmp.shape[0]
        df.append(df_tmp)
    df = pd.concat(df, ignore_index=True)
    logger.info("Fetched %s rows", rows)
    return df


###################################
def df_row_to_text(df, col_map):
    cols = [c for c in col_map if c in df.columns]

    return df[cols].apply(
        lambda row: " | ".join(
            f"{col_map[c]}: {row[c]}" for c in cols if pd.notna(row[c])
        ),
        axis=1,
    )


###################################
def clean_html(text):
    if pd.isna(text):
        return text

    soup = BeautifulSoup(text, "html.parser")
    return " ".join(soup.stripped_strings)
