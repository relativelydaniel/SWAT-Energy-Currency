import os
import sys
from pathlib import Path
from datacleanse import DataCleaner, Fetcher
from dotenv import load_dotenv
from engine import DataEngine, TradeExpScore
from config import COUNTRY_ISO_MAP

base_path = Path(__file__).resolve().parent
env_path = base_path / ".env"
load_dotenv(dotenv_path=env_path)


def trilateral_analysis():
    print("Initializing API fetch for Trilateral Analysis...") # functions per file - datacleanse.py
    api_url = os.environ.get("UNCOM_URL")
    api_key = os.environ.get("UNCOM_KEY")
    db_path = os.environ.get("DB_PATH")
    countries = COUNTRY_ISO_MAP

    print(f"DEBUG: URL found: {api_url is not None}")
    print(f"DEBUG: KEY found: {api_key is not None}")

    if api_url is None or api_key is None or db_path is None:
        print("CRITICAL ERROR: .env file elements missing")
        return

    cleaner = DataCleaner(db_path=db_path, api_url=api_url, api_key=api_key, countries=countries)
    fetcher = Fetcher(db_path=db_path)
    engine = DataEngine(cleaner=cleaner, fetcher=fetcher)

    try:
        # Engine Orchestration
        engine.sync_matrix(countries)
        engine.run_stats()
        engine.run_corr()
        return engine, cleaner, fetcher
    except Exception as e:
        print(f"Analytical Engine Pipeline failed: {e}")
        return False

def model_analysis(df, engine):
    print("\nCommencing Trade Exposure Score analysis...")
    if "gdp" not in df.columns:
        print("TES not calculated: GDP data is not yet available in the analysis table.")
        return engine, None

    analysis_df = df.copy()
    if "year" not in analysis_df.columns and "refyear" in analysis_df.columns:
        analysis_df = analysis_df.rename(columns={"refyear": "year"})
    if "year" not in analysis_df.columns:
        raise ValueError("Analysis data must contain either 'year' or 'refyear'.")
    gdp_df = analysis_df[["iso", "year", "gdp"]].drop_duplicates()
    tes_results = TradeExpScore(analysis_df).calculate_tes(gdp_df)
    return engine, tes_results

def run_swat():
    
    print("\nHello, and welcome to SWAT: a computational demonstration of the trilateral relationship of China, Nigeria, and Ghana.")
    result = trilateral_analysis()
    if not result:
        print("\nSWAT Fatal: Application dashboard execution halted due to engine synchronization failures.")
        return

    # Unpack the tuple returned by trilateral_analysis()
    engine, cleaner, fetcher = result

    df = engine.df
    engine, tes_results = model_analysis(df, engine)

    if tes_results is None:
        print("\nTES analysis is pending GDP ingestion.")
        return
    else:
        print("\nSWAT Success: Trade Exposure Score analysis completed.")
        print(tes_results.to_string(index=False))
        
if __name__ == "__main__":
    try:
        run_swat()
    except Exception as e:
        print(f"\nUnhandled Error: {e}")
    finally:
        print("\n" + "=" * 50)
        input("Press [ENTER] to close debugger...")