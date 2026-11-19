# test.py
import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock
from unittest.mock import Mock
from datacleanse import DataCleaner, Fetcher
from engine import DataEngine, TradeExpScore

@pytest.fixture
def mock_env_variables():
    # Provides fake configuration strings for class initializations
    return {
        "db_path": "sqlite:///:memory:",  
        "api_url": "https://api.comtrade.un.org/v1",
        "api_key": "fake_key_123",
        "countries": [288, 566, 156]
    }

@pytest.fixture
def sample_uncom_df():
    # Generates mock UN Comtrade data with pre-lowercased headers.
    return pd.DataFrame({
        'reportercode': [156, 566, 288],
        'refyear': [2022, 2022, 2022],
        'primaryvalue': [500000.0, np.nan, 120000.0],
        'custom_header_space': [1, 2, 3]
    })

@pytest.fixture
def sample_dbnomics_df():
    # Generates mock DBNomics data matching intermediate engine structure.
    return pd.DataFrame({
        'period': ['2022', '2022', '2022', '2022'],
        'value': [5.4, 14.2, 4.15, 195.5],
        'series_code': ['WB.WDI.FP.CPI.TOTL.ZG.CHN', 'WB.WDI.FP.CPI.TOTL.ZG.NGA', 'IMF.IFS.ENDE_XDC_USD_RATE.GH', 'IMF.IFS.ENDE_XDC_USD_RATE.NG'],
        'year': [2022, 2022, 2022, 2022],
        'type': ['inflation', 'inflation', 'exchange_rate', 'exchange_rate'] # Critical for pivot table operation
    })


def test_column_standardisation(mock_env_variables):
    # Verifies that column strings are formatted consistently across providers
    cleaner = DataCleaner(db_path=mock_env_variables["db_path"])
    cleaner.df = pd.DataFrame(columns=['Ref-Year', 'Trade Value (USD)', 'Gross–Wgt'])
    
    cleaner.standardise_columns()
    
    expected_cols = ['ref_year', 'trade_value_(usd)', 'gross_wgt']
    assert list(cleaner.df.columns) == expected_cols


def test_comtrade_fetch_uses_china_as_partner(mock_env_variables, monkeypatch):
    cleaner = DataCleaner(
        db_path=mock_env_variables["db_path"],
        api_url=mock_env_variables["api_url"],
        api_key=mock_env_variables["api_key"],
    )
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "dataset": [{"refYear": 2022, "partnerCode": 156, "primaryValue": 10.0}]
    }
    request = Mock(return_value=response)
    monkeypatch.setattr("datacleanse.requests.get", request)
    monkeypatch.setattr("datacleanse.time.sleep", lambda _: None)

    cleaner.fetch_api({288: "GHA", 566: "NGA"})

    assert request.call_count == 2
    assert {
        call.kwargs["params"]["reporterCode"]
        for call in request.call_args_list
    } == {"288", "566"}
    assert all(
        call.kwargs["params"]["partnerCode"] == "156"
        for call in request.call_args_list
    )
    assert cleaner.df["partnercode"].eq(156).all()


@pytest.mark.parametrize(
    "cached_df, accepted",
    [
        (pd.DataFrame({"partnercode": [156]}), True),
        (pd.DataFrame({"partnercode": [0]}), False),
        (pd.DataFrame({"partnercode": [156, 0]}), False),
        (pd.DataFrame({"partnercode": [156, np.nan]}), False),
        (pd.DataFrame({"reportercode": [288]}), False),
    ],
)
def test_fallback_requires_confirmed_china_partner(
    mock_env_variables, monkeypatch, cached_df, accepted
):
    cleaner = DataCleaner(
        db_path=mock_env_variables["db_path"],
        api_url=mock_env_variables["api_url"],
        api_key=mock_env_variables["api_key"],
    )
    response = Mock(status_code=503)
    monkeypatch.setattr("datacleanse.requests.get", Mock(return_value=response))
    monkeypatch.setattr("datacleanse.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "datacleanse.pd.read_sql_table", Mock(return_value=cached_df.copy())
    )

    if accepted:
        result = cleaner.fetch_api({288: "GHA"})
        assert cleaner.is_from_fallback
        assert result["partnercode"].eq(156).all()
    else:
        with pytest.raises(RuntimeError, match="not confirmed"):
            cleaner.fetch_api({288: "GHA"})
        assert not cleaner.is_from_fallback


def test_trade_exposure_score_calculates_country_energy_year_tes():
    trade = pd.DataFrame(
        {
            "iso": ["GHA", "GHA", "GHA", "GHA", "NGA"],
            "refyear": [2022, 2022, 2022, 2022, 2022],
            "partnercode": [156, 156, 156, 0, 156],
            "cmdcode": ["854143", "854143", "271600", "854143", "271600"],
            "flowcode": ["M", "X", "M", "M", "X"],
            "primaryvalue": [20.0, 10.0, 5.0, 900.0, 8.0],
        }
    )
    gdp = pd.DataFrame(
        {
            "iso": ["GHA", "NGA"],
            "year": [2022, 2022],
            "gdp": [1000.0, 400.0],
        }
    )

    result = TradeExpScore(trade).calculate_tes(gdp)

    solar = result.query("iso == 'GHA' and energy_type == 'solar_pv'").iloc[0]
    ghana_electricity = result.query(
        "iso == 'GHA' and energy_type == 'electricity'"
    ).iloc[0]
    nigeria_electricity = result.query(
        "iso == 'NGA' and energy_type == 'electricity'"
    ).iloc[0]
    assert solar["imports"] == 20.0
    assert solar["exports"] == 10.0
    assert solar["total_trade"] == 30.0
    assert solar["tes"] == 3.0
    assert ghana_electricity["tes"] == 0.5
    assert nigeria_electricity["tes"] == 2.0
    assert len(result) == 3


def test_trade_exposure_score_requires_gdp_for_each_country_year():
    trade = pd.DataFrame(
        {
            "iso": ["GHA"],
            "year": [2022],
            "partnercode": [156],
            "cmdcode": ["854143"],
            "flowcode": ["M"],
            "primaryvalue": [20.0],
        }
    )
    gdp = pd.DataFrame({"iso": ["NGA"], "year": [2022], "gdp": [1000.0]})

    with pytest.raises(ValueError, match="GDP is missing"):
        TradeExpScore(trade).calculate_tes(gdp)


def test_nans(mock_env_variables, sample_uncom_df):
    # Ensures numeric NaNs turn to zero and specified columns are safely dropped.
    cleaner = DataCleaner(db_path=mock_env_variables["db_path"])
    cleaner.df = sample_uncom_df
    
    cleaner.standardise_columns() # Essential prefix to lowercase columns
    cleaner.clean_data()
    
    # Assert missing numeric properties are filled with 0
    assert cleaner.df['primaryvalue'].isnull().sum() == 0
    assert cleaner.df['primaryvalue'].iloc[1] == 0
    
    # Assert specific metadata items are excluded
    assert 'refyear' not in cleaner.df.columns
    
def test_engine_sync(mock_env_variables, sample_uncom_df, sample_dbnomics_df):
    # Validates that the engine correctly merges UN Comtrade and DBNomics data into a unified DataFrame.
    cleaner = DataCleaner(db_path=mock_env_variables["db_path"])
    cleaner.df = sample_uncom_df
    cleaner.standardise_columns()
    cleaner.clean_data()
    
    fetcher = Fetcher(db_path=mock_env_variables["db_path"])
    fetcher.df = sample_dbnomics_df
    
    engine = DataEngine(cleaner=cleaner, fetcher=fetcher)
    
    # Mock the sync_matrix method to simulate successful merging
    engine.sync_matrix = MagicMock(return_value=True)
    
    result = engine.sync_matrix(mock_env_variables["countries"])
    
    assert result is True
    assert isinstance(engine.df, pd.DataFrame)
