#!/usr/bin/env python3
import pandas as pd
import fire


def get_categories(clini_excel: str, *columns: str):
    """Read all categories for a target label.

    Args:
        clini_excel:  Clini table to read from.
        columns:  target labels to determine categories for.
    """
    # clini_file = '/run/media/jxiaofeng/Sirius_02_empty/iCMS_project/tables/TCGA-CRC-DX_CLINI_new.xlsx'
    # columns = ['iCMS', 'CMS']
    df = pd.read_excel(clini_excel, dtype=str).dropna()

    print("{")
    for column in columns:
        print(f"    {column!r}: {list(df[column].unique())!r},")
    print("}")


if __name__ == "__main__":
    fire.Fire(get_categories)
