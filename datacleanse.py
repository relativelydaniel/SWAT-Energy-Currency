import pandas as pd
import numpy as np
from dbnomics import fetch_series
from wbgapi import wb 
import requests
import json
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from config import COMTRADE_PARTNER_CODE
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression 

class DataCleaner:
    def __init__(self, db_path=None, api_url=None, api_key=None, countries=None):
        self.engine = create_engine(db_path) if db_path else None
        self.api_key = api_key
        self.api_url = api_url
        self.db_path = db_path
        self.name = "bilateral_trade"
        self.df = pd.DataFrame()
        self.countries = countries
        self.is_from_fallback = False

    def fetch_api(self, countries):
        print("Executing API ingestion from UNCOM Trade...")
        frames = []
        fetch_errors = []
        max_attempts = 3
        for country in countries:
            params = {
                "reporterCode": str(country), 
                "partnerCode": str(COMTRADE_PARTNER_CODE),
                "period": "2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024",
                "cmdCode": "854143,271600" # solar, electricity
            } 
            # soon will subsitute with the params in CONFIG file
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            
            country_data = None
            for attempt in range(max_attempts):
                try:
                    response = requests.get(
                        self.api_url, params=params, headers=headers, timeout=30
                    )
                    if response.status_code == 429:
                        if attempt == max_attempts - 1:
                            fetch_errors.append(
                                f"{country}: rate limit persisted after {max_attempts} attempts"
                            )
                            break
                        retry_after = response.headers.get("Retry-After")
                        delay = 1.0
                        if retry_after:
                            try:
                                delay = max(0.0, float(retry_after))
                            except (TypeError, ValueError):
                                try:
                                    retry_time = parsedate_to_datetime(retry_after)
                                    if retry_time.tzinfo is None:
                                        retry_time = retry_time.replace(
                                            tzinfo=timezone.utc
                                        )
                                    delay = max(
                                        0.0,
                                        (
                                            retry_time - datetime.now(timezone.utc)
                                        ).total_seconds(),
                                    )
                                except (TypeError, ValueError, OverflowError):
                                    delay = 1.0
                        time.sleep(min(delay, 60.0))
                        continue
                    if response.status_code == 200:
                        result = response.json()
                        data = result.get("dataset") or result.get("data")
                        if data:
                            country_data = pd.json_normalize(data)
                        else:
                            fetch_errors.append(f"{country}: API returned no data")
                        break
                    else:
                        fetch_errors.append(f"{country}: HTTP {response.status_code}")
                        break
                except Exception as e:
                    fetch_errors.append(f"{country}: {e}")
                    break
            if country_data is not None:
                country_iso_map = {288: "GHA", 566: "NGA", 156: "CHN"}
                country_data["iso"] = country_iso_map.get(int(country), "UNKNOWN")
                frames.append(country_data)
                print(f"Successfully fetched {country}")
            time.sleep(1.1)

        if frames and not fetch_errors:
            self.is_from_fallback = False
            self.df = pd.concat(frames, ignore_index=True)
            self.standardise_columns()
            self.clean_data()
            return self.df
        else:
            try:
                if fetch_errors:
                    print(
                        "Warning: Comtrade fetch incomplete; discarding partial API "
                        f"results and checking database fallback. Details: {fetch_errors}"
                    )
                with self.engine.connect() as conn:
                    self.df = pd.read_sql_table(
                        self.name, 
                        con=conn, 
                        schema='Trade Intelligence')
                    if self.df.empty:
                        raise ValueError("No existing data available in the database.")
                    if 'partnercode' not in self.df.columns:
                        raise ValueError(
                            "Cached trade data has no partnercode; its partner scope cannot be confirmed."
                        )
                    if "reportercode" not in self.df.columns:
                        raise ValueError(
                            "Cached trade data has no reportercode; its country coverage cannot be confirmed."
                        )
                    partner_codes = pd.to_numeric(
                        self.df['partnercode'], errors='coerce'
                    )
                    if partner_codes.isna().any() or not partner_codes.eq(
                        COMTRADE_PARTNER_CODE
                    ).all():
                        raise ValueError(
                            f"Cached trade data is not confirmed as China-partner data "
                            f"(partner code {COMTRADE_PARTNER_CODE})."
                        )
                    reporter_codes = pd.to_numeric(
                        self.df["reportercode"], errors="coerce"
                    )
                    requested_reporters = {int(country) for country in countries}
                    if (
                        reporter_codes.isna().any()
                        or not requested_reporters.issubset(set(reporter_codes))
                    ):
                        raise ValueError(
                            "Cached trade data is missing one or more requested reporters; "
                            "country coverage cannot be confirmed."
                        )
                    print("Warning: API fetch failed. Using China-partner database data.")
                    self.is_from_fallback = True
                    return self.df
            except Exception as db_error:
                print(f"Database fetch failed: {db_error}")
                raise RuntimeError(
                    f"Critical pipeline error: API fetch failed; cached trade data "
                    f"is unavailable or not confirmed for partner "
                    f"{COMTRADE_PARTNER_CODE}: {db_error}"
                ) from db_error
    
    def standardise_columns(self):
        if self.df is not None and not self.df.empty:
            self.df.columns = (
                self.df.columns.str.strip()
                .str.replace(' ', '_')
                .str.replace('–', '_')
                .str.replace('-', '_')
                .str.lower()
            )
        return self.df

    def clean_data(self):
        print("\n--- Data Audit ft. First 10 rows ---")
        print(f"Initial Dimensions: {self.df.shape}")
        print(self.df.head())
        print(self.df.isnull().sum().reset_index(name = 'Missing Values Counted'))
        
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        print(self.df.info())
        
        metadata_cols = [
            'refperiodid', 'refmonth', 'partner2code',
            'motcode', 'qtyunitcode', 'altqtyunitcode', 'legacyestimationflag'
        ]
        
        self.df = self.df.drop(columns=metadata_cols, errors='ignore') # removes redundant metadata columns
        self.df = self.df.dropna(axis=1, how='all') # also drops columns that are entirely NaN
        self.df = self.df.loc[:, (self.df != 0).any(axis=0)] 
        print(f"Post Audit and Clean: {self.df.shape}")
        
        return self.df


    def connect_database(self, db_path = None):
        if self.df is None or self.df.empty:
            print("No data to push.")
            return
        
        active_engine = create_engine(db_path) if db_path else self.engine
        if not active_engine:
            print(f"CRITICAL ERROR: No database engine configured for '{self.name}'.")
            return
        
        try:
            with active_engine.begin() as conn:
                self.df.to_sql(
                    name=self.name, 
                    con=conn, 
                    schema='Trade Intelligence', 
                    if_exists="replace", 
                    index=False
                )
            print(f"Success: Table '{self.name}' written to database.")
        except Exception as e:
            print(f"CRITICAL ERROR during database load: {e}")


class Fetcher(): 
    def __init__(self, db_path):
        self.db_path = db_path
        self.engine = create_engine(db_path) if db_path else None
        self.name = "currency_stability"  
        self.df = pd.DataFrame()
        self.is_from_fallback = False

        
    def fetch_all(self): 
        print(f"Executing batch ingestion from WB + DB Nomics...")
        # set period 2014-24
        try:
            for country in str(COMTRADE_PARTNER_CODE).split(','):
                wb_df = wb.series.info(['FP.CPI.TOTL.ZG']) # fetch from World Bank WDI dataset for inflation per consumer price 
                gdp_df = wb.series.info(['NY.GDP.MKTP.CD']) # WDI dataset - current GDP
                curr_bal_df = wb.series.info(['BN.CAB.XOKA.CD']) # WDI dataset - current account balance to replace HFCE
            
            imf_df = fetch_series(provider_code='IMF', dataset_code='IFS',
                dimensions={'FREQ': ['A'], 
                            'REF_AREA': ['GH', 'NG', 'CN'], 
                            'INDICATOR': ['ENDE_XDC_USD_RATE']} # exchange rate in USD
            )
            
            
    
            
            if gdp_df is None or gdp_df.empty:
                raise ValueError("Required GDP API request returned no observations.")
            missing_columns = {"period", "value", "series_code"}.difference(
                gdp_df.columns
            )
            if missing_columns:
                raise ValueError(
                    f"GDP series is missing columns: {sorted(missing_columns)}"
                )
            gdp_df = gdp_df.copy()
            gdp_df["type"] = "gdp"
            gdp_df["year"] = pd.to_datetime(
                gdp_df["period"], errors="coerce"
            ).dt.year
            if "iso" not in gdp_df.columns:
                gdp_df["iso"] = np.nan
            code_iso = gdp_df["series_code"].astype(str).str.extract(
                r"(GHA|NGA|CHN)$", expand=False
            )
            gdp_df["iso"] = gdp_df["iso"].fillna(code_iso)
            if gdp_df[["year", "iso"]].isna().any().any():
                raise ValueError("GDP series has missing country or year keys.")
            gdp_df["iso"] = gdp_df["iso"].astype("string").str.strip().str.upper()
            # Keep DBnomics for NGA through 2018; use the current WB source afterward.
            gdp_df = gdp_df.loc([(
                    gdp_df["iso"].eq("NGA")
                    & gdp_df["year"].between(2019, 2024)
                )])
            
            gdp_df = pd.concat(
                [gdp_df], ignore_index=True
            )
            gdp_df["iso"] = gdp_df["iso"].astype("string").str.strip().str.upper()
            gdp_df["year"] = pd.to_numeric(gdp_df["year"], errors="coerce")
            nga_recent_gdp = gdp_df.loc[
                gdp_df["iso"].eq("NGA") & gdp_df["year"].between(2019, 2024)
            ]
            expected_nga_years = set(range(2019, 2025))
            invalid_nga_years = (
                nga_recent_gdp["year"].isna()
                | nga_recent_gdp["year"].mod(1).ne(0)
            )
            if (
                invalid_nga_years.any()
                or nga_recent_gdp["year"].duplicated().any()
                or set(nga_recent_gdp["year"].astype(int)) != expected_nga_years
            ):
                raise ValueError(
                    "NGA GDP must contain exactly one observation for each "
                    "integer year from 2019 through 2024."
                )
                        
            # Define Imputation Logic
            def _fill_group(group):
                if group["value"].isna().all() or not group["value"].isna().any():
                    return group

                known = group.dropna(subset=["value"])
                missing = group[group["value"].isna()]

                if len(known) < 2:
                    group["value"] = group["value"].ffill().bfill()
                    return group

                # Fit Model: Year -> Value
                X_train = known[["year"]].values
                y_train = known["value"].values

                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict missing years (NGA 2022-2024 & CHN 2023)
                X_miss = missing[["year"]].values
                predicted = model.predict(X_miss)

                # Assign predictions
                group.loc[group["value"].isna(), "value"] = predicted
                return group

            # Apply Imputation GROUPED BY ISO & SERIES_CODE
            hfce_df = hfce_df.groupby(["iso", "series_code"], group_keys=False).apply(
                _fill_group
            )

            if wb_df.empty and imf_df.empty and hfce_df.empty:
                raise Exception("WB-DBN returned an empty dataset for all providers.")
            fetched_df = pd.concat([wb_df, imf_df, curr_bal_df,  gdp_df], ignore_index=True) # sticks into master table

            df_cleaned = pd.DataFrame({
                "period": fetched_df["period"],
                "value": pd.to_numeric(fetched_df["value"], errors="coerce"),
                "series_code": fetched_df["series_code"],
            })

            def assign_type(series_code):
                code_upper = str(series_code).upper()
                if 'CPI' in code_upper or 'INFLATION' in code_upper:
                    return 'inflation'
                elif 'ENDE_XDC_USD_RATE' in code_upper or 'EXCHANGE' in code_upper:
                    return 'exchange_rate'
                elif 'GDP' in code_upper:
                    return 'gdp'
               # current account balance is used to replace HFCE
                elif 'BN.CAB.XOKA.CD' in code_upper or 'CURRENT_ACCOUNT_BALANCE' in code_upper:
                    return 'current_account_balance'
                else:
                    return 'unknown'
                
            inferred_type = df_cleaned["series_code"].apply(assign_type)
            if "type" in fetched_df.columns:
                df_cleaned["type"] = fetched_df["type"].fillna(inferred_type)
            else:
                df_cleaned["type"] = inferred_type

            if "year" in fetched_df.columns:
                df_cleaned["year"] = pd.to_numeric(
                    fetched_df["year"], errors="coerce"
                )
            else:
                df_cleaned["year"] = np.nan
            period_year = pd.to_datetime(
                df_cleaned["period"], errors="coerce"
            ).dt.year
            
            df_cleaned["year"] = df_cleaned["year"].fillna(period_year)
            if df_cleaned["year"].isna().any():
                raise ValueError("Fetched macro data contains rows without a valid year.")
            df_cleaned["year"] = df_cleaned["year"].astype(int)
            df_cleaned = df_cleaned[df_cleaned["year"].between(2014, 2024)]
            
            country_aliases = {
                "GHA": "GHA",
                "GH": "GHA",
                "GHANA": "GHA",
                "NGA": "NGA",
                "NG": "NGA",
                "NIGERIA": "NGA",
                "CHN": "CHN",
                "CN": "CHN",
                "CHINA": "CHN",
            }
            if "iso" in fetched_df.columns:
                raw_iso = (
                    fetched_df["iso"].astype("string").str.upper().str.strip()
                )
                df_cleaned["iso"] = raw_iso.map(country_aliases)
            else:
                df_cleaned["iso"] = np.nan
            for country_column in (
                "REF_AREA",
                "ref_area",
                "country_code",
                "country",
                "COUNTRY",
            ):
                if country_column in fetched_df.columns:
                    dimension_iso = fetched_df[country_column].astype(
                        "string"
                    ).str.upper().str.strip().map(country_aliases)
                    df_cleaned["iso"] = df_cleaned["iso"].fillna(dimension_iso)
            code_iso = df_cleaned["series_code"].astype(str).str.extract(
                r"(?:^|[-._])(GHA|NGA|CHN|GH|NG|CN)$", expand=False
            )
            code_iso = code_iso.replace(
                {"GH": "GHA", "NG": "NGA", "CN": "CHN"}
            )
            df_cleaned["iso"] = df_cleaned["iso"].fillna(code_iso)
            if df_cleaned["iso"].isna().any():
                unresolved = fetched_df.loc[
                    df_cleaned["iso"].isna(),
                    [
                        column
                        for column in (
                            "type",
                            "series_code",
                            "iso",
                            "REF_AREA",
                            "ref_area",
                            "country_code",
                            "country",
                        )
                        if column in fetched_df.columns
                    ],
                ].drop_duplicates()
                raise ValueError(
                    "Fetched macro data contains rows without a valid country "
                    f"code: {unresolved.to_dict(orient='records')}"
                )
            gdp_rows = df_cleaned["type"].eq("gdp")
            if not gdp_rows.any() or not df_cleaned.loc[gdp_rows, "value"].notna().any():
                raise ValueError("Required GDP series is absent from fetched macro data.")
            gdp_values = df_cleaned.loc[gdp_rows, "value"].dropna()
            if (gdp_values <= 0).any():
                raise ValueError(
                    "Required GDP series contains non-positive values."
                )

            self.df = df_cleaned
            self.is_from_fallback = False
            print(f"-> Successfully synchronized series from WB, IMF, and HCFE.")
            self.standardise_columns()
            self.clean_data()
            return self.df
        except Exception as e:
            # pull from existing PGSQL database from previous runs if API fails
            try:
                with self.engine.connect() as conn:
                    self.df = pd.read_sql_table(
                        self.name, 
                        con=conn, 
                        schema='Trade Intelligence')
                if not self.df.empty:
                        print("Warning: API fetch failed. Using existing database data.")
                        self.is_from_fallback = True
                        return self.df
                else:
                        raise Exception("No existing data available in the database.")
            except Exception as db_e:
                print(f"Database fetch failed: {db_e}")
                raise Exception(f"Critical pipeline error: {e}")
        
    def clean_data(self):
        print("\n--- Data Audit ft. First 10 rows ---")
        print(f"Initial Dimensions: {self.df.shape}")
        print(self.df.head())
        print(self.df.isnull().sum().reset_index(name = 'Missing Values Counted')) # extend a little
        
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        print(self.df.info())
        # Add this right after fetching DB Nomics data:
        print("DB Nomics Unique Data Types:", self.df['type'].unique())
        
        self.df = self.df.dropna(axis=1, how='all')
        self.df = self.df.loc[:, (self.df != 0).any(axis=0)]
        
    
        print(f"Post Audit and Clean: {self.df.shape}")
        return self.df
        
    def standardise_columns(self):
        if self.df is not None and not self.df.empty:
            self.df.columns = (
                self.df.columns.str.strip()
                .str.replace(' ', '_')
                .str.replace('–', '_')
                .str.replace('-', '_')
                .str.lower()
            )
        return self.df
    
    def connect_database(self, db_path = None): 
        if self.df is None or self.df.empty:
            print("No data to push.")
            return
        
        if getattr(self, 'is_from_fallback', False):
            print(f"Skipping DB-write for '{self.name}' (loaded from fallback)")
            return
        
        try:
            with self.engine.begin() as conn:
                self.df.to_sql(
                    name = self.name, 
                    con = conn, 
                    schema='Trade Intelligence', 
                    if_exists="replace", 
                    index=False
                )
            print(f"Success: Table '{self.name}' written to database.")
        except Exception as e:
            print(f"CRITICAL ERROR during database load: {e}")
        
    
 
    def json_dc(self): 
        return {
            "api_settings": {
                "uncom_base_url": "https://api.comtrade.un.org/v1",
                "cmd_codes": ["854143", "271600"],
                "periods": "2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024"
            },
            
            "countries": {
                "reporting": [288, 566],
                "iso_mapping": {
                    "288": "GHA",
                    "566": "NGA",
                    "156": "CHN"
                }
            },
            
            "pipeline": {
                "target_table": "bilateral_trade",
                "schema": "Trade Intelligence",
                "clean_nulls": True
            }
        }