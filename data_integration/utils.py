import pandas as pd
from datetime import datetime
import yaml
import os
import numpy as np
import hashlib
import re

CSV_PATH = "etl_system\etl_job.csv"  # đường dẫn file CSV
YML_PATH = "column_transform.yml"
def get_tables_to_sync():
    """Đọc danh sách job đang active từ file CSV"""
    try:
        df = pd.read_csv(CSV_PATH)
        df_active = df[df['ACTIVE'] == 1][['JOB_NAME', 'TARGET_TABLE', 'P_KEY', 'DATASOURCE_NUM']]
        return df_active.reset_index(drop=True)
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return pd.DataFrame()

def start_job(job_name):
    """Cập nhật trạng thái job thành đang chạy"""
    try:
        df = pd.read_csv(CSV_PATH)
        if job_name not in df['JOB_NAME'].values:
            print(f"Không tìm thấy job: {job_name}")
            return
        df.loc[df['JOB_NAME'] == job_name, 'START_TS'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df.loc[df['JOB_NAME'] == job_name, 'STATUS'] = -1
        df.to_csv(CSV_PATH, index=False)
    except Exception as e:
        print(f"Lỗi khi cập nhật job: {e}")

def end_job(job_name, error_list):
    """Cập nhật trạng thái job thành hoàn thành"""
    try:
        df = pd.read_csv(CSV_PATH)
        if job_name not in df['JOB_NAME'].values:
            print(f"Không tìm thấy job: {job_name}")
            return
        df.loc[df['JOB_NAME'] == job_name, 'END_TS'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df.loc[df['JOB_NAME'] == job_name, 'STATUS'] = 1
        df.loc[df['JOB_NAME'] == job_name, 'ERROR_MESSAGE'] = ", ".join(error_list)
        df.to_csv(CSV_PATH, index=False)
        print(f"Đã cập nhật END_TS và STATUS = 1 cho job: {job_name}")
    except Exception as e:
        print(f"Lỗi khi cập nhật job: {e}")

# ------------------------------
# 1️⃣ Đọc file cấu hình YAML
# ------------------------------
def read_yml_config():
    """
    Đọc file YAML chứa định nghĩa bảng (tables, columns, transform_*)
    Trả về list các dict dạng:
    [
      {
        "name": "blinkit_products",
        "columns": {"product_id": None, "product_name": None, ...},
        "transform_text": [...],
        "transform_null_identity": [...],
        "transform_null_val": [...]
      },
      ...
    ]
    """
    with open(YML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    result = []
    for table in data.get("tables", []):
        name = table.get("name")

        # Chuyển "columns" từ list -> dict
        columns_list = table.get("columns", [])
        columns_dict = {}
        for item in columns_list:
            if isinstance(item, dict):
                columns_dict.update(item)

        # Lấy các loại transform
        result.append({
            "name": name,
            "columns": columns_dict,
            "transform_text": table.get("transform_text", []),
            "transform_null_identity": table.get("transform_null_identity", []),
            "transform_null_val": table.get("transform_null_val", [])
        })

    return result


# ------------------------------
# 2️⃣ Áp dụng transform
# ------------------------------
def apply_transform(df: pd.DataFrame, table_name: str, config: list) -> pd.DataFrame:
    """
    Áp dụng transform theo cấu hình YAML cho bảng tương ứng.

    Parameters
    ----------
    df : pd.DataFrame
        Dữ liệu cần xử lý.
    table_name : str
        Tên bảng cần áp dụng transform.
    config : list
        Cấu hình được đọc từ read_yml_config().

    Returns
    -------
    pd.DataFrame
        DataFrame sau khi transform.
    """
    df = df.copy()
    table_conf = next((t for t in config if t["name"] == table_name), None)
    if table_conf is None:
        raise ValueError(f"Không tìm thấy bảng '{table_name}' trong config YAML.")

    # --- transform_text ---
    for transform in table_conf.get("transform_text", []):
        for col, expr in transform.items():
            if col not in df.columns:
                continue

            if expr.startswith("REGEXP_REPLACE"):
                # Dạng REGEXP_REPLACE({entity}, '[^a-zA-Z0-9 ]', '')
                pattern = re.findall(r"REGEXP_REPLACE\(\{entity\}, '([^']+)', '([^']*)'\)", expr)
                if pattern:
                    pat, repl = pattern[0]
                    df[col] = df[col].astype(str).apply(lambda x: re.sub(pat, repl, x))
            elif expr.startswith("UPPER"):
                df[col] = df[col].astype(str).str.upper()
            elif expr.startswith("LOWER"):
                df[col] = df[col].astype(str).str.lower()

    # --- transform_null_identity ---
    for transform in table_conf.get("transform_null_identity", []):
        for col, default_val in transform.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)

    # --- transform_null_val ---
    for transform in table_conf.get("transform_null_val", []):
        for col, method in transform.items():
            if col not in df.columns:
                continue
            if df[col].isna().sum() == 0:
                continue
            
            if "," not in method:
                if method.upper() == "MODE":
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                    df[col] = df[col].fillna(mode_val)
                elif method.upper() == "MEDIAN":
                    df[col] = df[col].fillna(df[col].median())
                elif method.upper() == "MEAN":
                    df[col] = df[col].fillna(df[col].mean())
            else:
                method_name = method.upper().split(",")[0]
                method_groupby = method.upper().split(",")[1].split("|")
                if method_name == "MEDIAN":
                    df[col] = df.groupby(method_groupby)[col].transform(lambda x: x.fillna(x.median()))
                if method_name == "MEAN":
                    df[col] = df.groupby(method_groupby)[col].transform(lambda x: x.fillna(x.mean()))
                if method_name == "MODE":
                    df[col] = df.groupby(method_groupby)[col].transform(
                        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan)
                    )
    return df


def add_key_column(df: pd.DataFrame, columns: list, constant: str = None, new_name_col: str = "NEW_COL") -> pd.DataFrame:
    """
    Tạo cột 'key' bằng cách băm (hash) các giá trị từ danh sách cột được truyền vào,
    cộng thêm biến hằng `constant`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đầu vào.
    columns : list
        Danh sách tên các cột cần kết hợp để tạo khóa.
    constant : str, optional
        Biến hằng được nối thêm trước khi băm (mặc định là 'CONST').
    new_name_col : str, optional
        Tên của cột mới được sinh ra (mặc định là 'CONST').

    Returns
    -------
    pd.DataFrame
        DataFrame có thêm cột 'key' (kiểu int32).
    """

    def hash_row(row):
        if constant:
            # Nối các giá trị cột + biến hằng thành chuỗi
            combined = constant + "_" + "_".join(str(row[col]) for col in columns)
        else: combined = "_".join(str(row[col]) for col in columns)
        # Tạo hash SHA256, rồi chuyển về int32
        hash_val = int(hashlib.sha256(combined.encode('utf-8')).hexdigest(), 16)
        return np.int32(hash_val % (2**31 - 1))  # ép về int32 để tránh tràn số

    df = df.copy()
    df[new_name_col] = df.apply(hash_row, axis=1)
    return df