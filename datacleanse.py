import pandas as pd
import numpy as np
from dbnomics import fetch_series
import wbgapi as wb 
import requests
import json
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from config import COMTRADE_PARTNER_CODE, COUNTRY_ISO_MAP
from sqlalchemy import create_engine

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
                    if isinstance(
                        e,
                        (
                            requests.exceptions.Timeout,
                            requests.exceptions.ConnectionError,
                        ),
                    ):
                        if attempt < max_attempts - 1:
                            delay = min(2**attempt, 8)
                            print(
                                f"Warning: Comtrade request for {country} "
                                f"failed ({e}); retrying in {delay}s "
                                f"({attempt + 2}/{max_attempts})."
                            )
                            time.sleep(delay)
                            continue
                        fetch_errors.append(
                            f"{country}: request failed after {max_attempts} "
                            f"attempts: {e}"
                        )
                    else:
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
                    f"{COMTRADE_PARTNER_CODE}. API errors: {fetch_errors}. "
                    f"Cache error: {db_error}"
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
        try:
            wb_indicators = {
                "FP.CPI.TOTL.ZG": "inflation",
                "NY.GDP.MKTP.CD": "gdp",
                "BN.CAB.XOKA.CD": "current_account_balance",
            }
            wb_frames = []
            for country in COUNTRY_ISO_MAP.values():
                country_df = wb.data.DataFrame(
                    list(wb_indicators),
                    economy=country,
                    time=range(2014, 2025),
                    columns="series",
                ).reset_index()
                if "time" not in country_df.columns:
                    raise ValueError(
                        f"World Bank data for {country} has no time dimension."
                    )
                country_df = country_df.melt(
                    id_vars=["time"],
                    value_vars=list(wb_indicators),
                    var_name="series_code",
                    value_name="value",
                )
                country_df["year"] = pd.to_numeric(
                    country_df["time"].astype(str).str.extract(r"(\d{4})")[0],
                    errors="coerce",
                )
                if country_df["year"].isna().any():
                    raise ValueError(
                        f"World Bank data for {country} contains invalid years."
                    )
                country_df["period"] = pd.to_datetime(
                    country_df["year"].astype(int).astype(str), format="%Y"
                )
                country_df["iso"] = country
                country_df["type"] = country_df["series_code"].map(wb_indicators)
                wb_frames.append(country_df[
                    ["period", "value", "series_code", "type", "year", "iso"]
                ])
            wb_df = pd.concat(wb_frames, ignore_index=True)

            imf_df = fetch_series(provider_code='IMF', dataset_code='IFS',
                dimensions={'FREQ': ['A'], 
                            'REF_AREA': ['GH', 'NG'], 
                            'INDICATOR': ['ENDE_XDC_USD_RATE']} # exchange rate in USD
            )
            if imf_df is None or imf_df.empty:
                raise ValueError("Required IMF exchange-rate request returned no data.")
            if "period" not in imf_df.columns or "value" not in imf_df.columns:
                raise ValueError("IMF data must contain period and value columns.")
            imf_df = imf_df.copy()
            imf_df["series_code"] = imf_df.get(
                "series_code", "IMF.IFS.ENDE_XDC_USD_RATE"
            )
            imf_df["type"] = "exchange_rate"
            imf_df["year"] = pd.to_datetime(
                imf_df["period"], errors="coerce"
            ).dt.year
            country_aliases = {
                "GHA": "GHA", "GH": "GHA", "GHANA": "GHA",
                "NGA": "NGA", "NG": "NGA", "NIGERIA": "NGA",
            }
            imf_df["iso"] = np.nan
            for country_column in ("iso", "REF_AREA", "ref_area", "country"):
                if country_column in imf_df.columns:
                    values = (
                        imf_df[country_column]
                        .astype("string")
                        .str.upper()
                        .str.strip()
                        .map(country_aliases)
                    )
                    imf_df["iso"] = imf_df["iso"].fillna(values)
            code_iso = imf_df["series_code"].astype(str).str.extract(
                r"(?:^|[-._])(GHA|NGA|GH|NG)$", expand=False
            )
            imf_df["iso"] = imf_df["iso"].fillna(
                code_iso.replace({"GH": "GHA", "NG": "NGA"})
            )
            if imf_df[["year", "iso"]].isna().any().any():
                raise ValueError("IMF data has missing country or year keys.")

            fetched_df = pd.concat(
                [
                    wb_df,
                    imf_df[["period", "value", "series_code", "type", "year", "iso"]],
                ],
                ignore_index=True,
            )
            fetched_df["value"] = pd.to_numeric(
                fetched_df["value"], errors="coerce"
            )
            fetched_df["year"] = pd.to_numeric(
                fetched_df["year"], errors="coerce"
            )
            fetched_df = fetched_df.loc[
                fetched_df["year"].between(2014, 2024)
            ].copy()
            if fetched_df.empty:
                raise ValueError("World Bank and IMF returned no data for 2014–2024.")
            gdp_rows = fetched_df["type"].eq("gdp")
            if not gdp_rows.any() or not fetched_df.loc[gdp_rows, "value"].gt(0).any():
                raise ValueError("Required GDP series has no positive observations.")

            self.df = fetched_df
            self.is_from_fallback = False
            self.standardise_columns()
            self.clean_data()
            print("-> Successfully synchronized World Bank and IMF series.")
            return self.df
        except Exception as e:
            try:
                with self.engine.connect() as conn:
                    cached = pd.read_sql_table(
                        self.name, con=conn, schema="Trade Intelligence"
                    )
                if cached.empty:
                    raise ValueError("No existing macro data available in the database.")
                required_columns = {"iso", "year", "type", "value"}
                if not required_columns.issubset(cached.columns):
                    raise ValueError(
                        "Cached macro data is missing required country-year fields."
                    )
                cached["iso"] = cached["iso"].astype("string").str.strip().str.upper()
                cached["year"] = pd.to_numeric(cached["year"], errors="coerce")
                cached["value"] = pd.to_numeric(cached["value"], errors="coerce")
                cached_gdp = cached.loc[cached["type"].eq("gdp")]
                if (
                    cached[["iso", "year"]].isna().any().any()
                    or cached_gdp.empty
                    or not cached_gdp["value"].gt(0).any()
                ):
                    raise ValueError(
                        "Cached macro data has no valid country-year GDP observations."
                    )
                self.df = cached
                self.standardise_columns()
                self.is_from_fallback = True
                print("Warning: API fetch failed. Using existing database data.")
                return self.df
            except Exception as db_error:
                print(f"Database fetch failed: {db_error}")
                raise RuntimeError(
                    f"Critical macro-data ingestion error: {e}; "
                    f"database fallback failed: {db_error}"
                ) from e
        
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