# test.py
import pytest
import pandas as pd
import numpy as np
import os
import requests
from unittest.mock import MagicMock
from unittest.mock import Mock
from datacleanse import DataCleaner, Fetcher
from engine import DataEngine, TradeExpScore
from main import model_analysis


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


def test_comtrade_429_retries_using_retry_after(mock_env_variables, monkeypatch):
    cleaner = DataCleaner(
        db_path=mock_env_variables["db_path"],
        api_url=mock_env_variables["api_url"],
        api_key=mock_env_variables["api_key"],
    )
    rate_limited = Mock(status_code=429, headers={"Retry-After": "0.25"})
    success = Mock(status_code=200, headers={})
    success.json.return_value = {
        "dataset": [{"partnerCode": 156, "primaryValue": 10.0}]
    }
    request = Mock(side_effect=[rate_limited, success])
    sleep = Mock()
    monkeypatch.setattr("datacleanse.requests.get", request)
    monkeypatch.setattr("datacleanse.time.sleep", sleep)

    cleaner.fetch_api({288: "GHA"})

    assert request.call_count == 2
    sleep.assert_any_call(0.25)
    assert cleaner.df["primaryvalue"].tolist() == [10.0]


def test_comtrade_timeout_retries_then_succeeds(mock_env_variables, monkeypatch):
    cleaner = DataCleaner(
        db_path=mock_env_variables["db_path"],
        api_url=mock_env_variables["api_url"],
        api_key=mock_env_variables["api_key"],
    )
    success = Mock(status_code=200, headers={})
    success.json.return_value = {
        "dataset": [{"partnerCode": 156, "primaryValue": 10.0}]
    }
    request = Mock(
        side_effect=[
            requests.exceptions.ReadTimeout("read timed out"),
            success,
        ]
    )
    sleep = Mock()
    monkeypatch.setattr("datacleanse.requests.get", request)
    monkeypatch.setattr("datacleanse.time.sleep", sleep)

    result = cleaner.fetch_api({288: "GHA"})

    assert request.call_count == 2
    sleep.assert_any_call(1)
    assert result["primaryvalue"].tolist() == [10.0]
    assert not cleaner.is_from_fallback


def test_comtrade_timeout_exhaustion_does_not_accept_partial_results(
    mock_env_variables, monkeypatch
):
    cleaner = DataCleaner(
        db_path=mock_env_variables["db_path"],
        api_url=mock_env_variables["api_url"],
        api_key=mock_env_variables["api_key"],
    )
    request = Mock(
        side_effect=requests.exceptions.ReadTimeout("read timed out")
    )
    monkeypatch.setattr("datacleanse.requests.get", request)
    monkeypatch.setattr("datacleanse.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "datacleanse.pd.read_sql_table",
        Mock(side_effect=ValueError("Table bilateral_trade not found")),
    )

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        cleaner.fetch_api({288: "GHA"})

    assert request.call_count == 3
    assert cleaner.df.empty
    assert not cleaner.is_from_fallback


def test_partial_comtrade_fetch_uses_cache_instead_of_partial_rows(
    mock_env_variables, monkeypatch
):
    cleaner = DataCleaner(
        db_path=mock_env_variables["db_path"],
        api_url=mock_env_variables["api_url"],
        api_key=mock_env_variables["api_key"],
    )
    success = Mock(status_code=200, headers={})
    success.json.return_value = {
        "dataset": [{"partnerCode": 156, "primaryValue": 10.0}]
    }
    no_data = Mock(status_code=200, headers={})
    no_data.json.return_value = {"dataset": []}
    request = Mock(side_effect=[success, no_data])
    cached = pd.DataFrame(
        {
            "partnercode": [156, 156],
            "reportercode": [288, 566],
            "primaryvalue": [33.0, 44.0],
        }
    )
    monkeypatch.setattr("datacleanse.requests.get", request)
    monkeypatch.setattr("datacleanse.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "datacleanse.pd.read_sql_table", Mock(return_value=cached.copy())
    )

    result = cleaner.fetch_api({288: "GHA", 566: "NGA"})

    assert cleaner.is_from_fallback
    assert result["primaryvalue"].tolist() == [33.0, 44.0]


@pytest.mark.parametrize(
    "cached_df, accepted",
    [
        (pd.DataFrame({"partnercode": [156], "reportercode": [288]}), True),
        (pd.DataFrame({"partnercode": [0], "reportercode": [288]}), False),
        (pd.DataFrame({"partnercode": [156, 0], "reportercode": [288, 288]}), False),
        (pd.DataFrame({"partnercode": [156, np.nan], "reportercode": [288, 288]}), False),
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
            "iso": ["GHA", "GHA", "GHA", "GHA", "NGA", "GHA", "GHA"],
            "refyear": [2022, 2022, 2022, 2022, 2022, 2022, 2022],
            "partnercode": [156, 156, 156, 0, 156, 156, 156],
            "cmdcode": [
                854143.0,
                854143.0,
                271600.0,
                854143.0,
                271600.0,
                "854143.0000000000000000001",
                "271600.0000000000000000001",
            ],
            "flowcode": ["M", "X", "M", "M", "X", "M", "M"],
            "primaryvalue": [20.0, 10.0, 5.0, 900.0, 8.0, 500.0, 600.0],
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


def test_trade_exposure_score_ignores_component_flows_when_total_flows_exist():
    trade = pd.DataFrame(
        {
            "iso": ["GHA", "GHA"],
            "year": [2022, 2022],
            "partnercode": [156, 156],
            "cmdcode": ["854143", "854143"],
            "flowcode": ["M", "FM"],
            "primaryvalue": [20.0, 7.0],
        }
    )
    gdp = pd.DataFrame({"iso": ["GHA"], "year": [2022], "gdp": [1000.0]})

    result = TradeExpScore(trade).calculate_tes(gdp)

    assert result.loc[0, "imports"] == 20.0
    assert result.loc[0, "exports"] == 0.0
    assert result.loc[0, "total_trade"] == 20.0


def test_trade_exposure_score_requires_gdp_for_each_trade_observation():
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

    with pytest.raises(ValueError, match="GDP is required.*GHA"):
        TradeExpScore(trade).calculate_tes(gdp)


def test_trade_exposure_score_rejects_non_numeric_gdp():
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
    gdp = pd.DataFrame({"iso": ["GHA"], "year": [2022], "gdp": ["not-available"]})

    with pytest.raises(ValueError, match="non-numeric GDP"):
        TradeExpScore(trade).calculate_tes(gdp)


def test_trade_exposure_score_rejects_missing_gdp_for_any_scored_year():
    trade = pd.DataFrame(
        {
            "iso": ["GHA", "GHA"],
            "year": [2022, 2023],
            "partnercode": [156, 156],
            "cmdcode": ["854143", "854143"],
            "flowcode": ["M", "M"],
            "primaryvalue": [20.0, 30.0],
        }
    )
    gdp = pd.DataFrame(
        {
            "iso": ["GHA", "GHA", "NGA"],
            "year": [2022, 2023, 2022],
            "gdp": [1000.0, np.nan, 500.0],
        }
    )

    with pytest.raises(ValueError, match="GDP is required.*2023"):
        TradeExpScore(trade).calculate_tes(gdp)


def test_model_analysis_accepts_refyear_only_trade_frame():
    df = pd.DataFrame(
        {
            "iso": ["GHA"],
            "refyear": [2022],
            "partnercode": [156],
            "cmdcode": [854143.0],
            "flowcode": ["M"],
            "primaryvalue": [20.0],
            "gdp": [1000.0],
        }
    )

    _, result = model_analysis(df, engine=None)

    assert result.loc[0, "year"] == 2022
    assert result.loc[0, "tes"] == 2.0


def test_fetch_all_includes_normalized_gdp_rows(mock_env_variables, monkeypatch):
    indicators = [
        "FP.CPI.TOTL.ZG",
        "NY.GDP.MKTP.CD",
        "BN.CAB.XOKA.CD",
    ]
    years = [f"YR{year}" for year in range(2014, 2025)]

    def world_bank_frame(series, economy, time, columns):
        assert series == indicators
        assert time == range(2014, 2025)
        assert columns == "series"
        values = {
            "FP.CPI.TOTL.ZG": [5.0] * len(years),
            "NY.GDP.MKTP.CD": [2_000.0] * len(years),
            "BN.CAB.XOKA.CD": [100.0] * len(years),
        }
        return pd.DataFrame(
            values,
            index=pd.MultiIndex.from_product(
                [[economy], years], names=["economy", "time"]
            ),
        )

    wb_request = Mock(side_effect=world_bank_frame)
    monkeypatch.setattr("datacleanse.wb.data.DataFrame", wb_request)
    imf_data = pd.DataFrame(
        {
            "period": ["2022", "2022"],
            "value": [10.0, 20.0],
            "series_code": ["IMF.IFS.ENDE_XDC_USD_RATE"] * 2,
            "REF_AREA": ["GH", "NG"],
        }
    )
    fetch_mock = Mock(return_value=imf_data)
    monkeypatch.setattr("datacleanse.fetch_series", fetch_mock)
    fetcher = Fetcher(db_path=mock_env_variables["db_path"])

    result = fetcher.fetch_all()

    gdp = result.query("type == 'gdp' and iso == 'GHA'").iloc[0]
    assert gdp["year"] == 2014
    assert gdp["value"] == 2000.0
    exchange_rate = result.query("type == 'exchange_rate' and iso == 'GHA'").iloc[0]
    assert exchange_rate["value"] == 10.0
    assert result.query(
        "type == 'current_account_balance' and iso == 'NGA'"
    ).iloc[0]["value"] == 100.0
    assert wb_request.call_count == 2
    assert [call.kwargs["economy"] for call in wb_request.call_args_list] == [
        "GHA",
        "NGA",
    ]
    fetch_mock.assert_called_once()


def test_fetch_all_rejects_empty_world_bank_data_without_cache(
    mock_env_variables, monkeypatch
):
    wb_request = Mock(side_effect=Exception("unexpected failure"))
    monkeypatch.setattr("datacleanse.wb.data.DataFrame", wb_request)
    fetcher = Fetcher(db_path=mock_env_variables["db_path"])

    with pytest.raises(RuntimeError, match="database fallback failed"):
        fetcher.fetch_all()


def test_gdp_observation_survives_engine_pivot_and_trade_join():
    class MockCleaner:
        def fetch_api(self, countries):
            return pd.DataFrame(
                {
                    "iso": ["GHA"],
                    "refyear": [2022],
                    "partnercode": [156],
                    "cmdcode": ["854143"],
                    "flowcode": ["M"],
                    "primaryvalue": [50.0],
                }
            )

        def connect_database(self):
            pass

    class MockFetcher:
        def fetch_all(self):
            return pd.DataFrame(
                {
                    "iso": ["GHA", "GHA"],
                    "year": [2022, 2022],
                    "type": ["gdp", "inflation"],
                    "value": [2000.0, 5.0],
                }
            )

        def connect_database(self):
            pass

    engine = DataEngine(cleaner=MockCleaner(), fetcher=MockFetcher())
    engine.sync_matrix({288: "GHA"})

    matching = engine.df.query("iso == 'GHA' and year == 2022")
    assert len(matching) == 1
    assert matching.iloc[0]["gdp"] == 2000.0
    assert matching.iloc[0]["inflation"] == 5.0


def test_current_account_balance_reaches_matching_trade_rows():
    class MockCleaner:
        def fetch_api(self, countries):
            return pd.DataFrame(
                {
                    "iso": ["NGA"],
                    "refyear": [2014],
                    "partnercode": [156],
                    "cmdcode": ["854143"],
                    "flowcode": ["M"],
                    "primaryvalue": [50.0],
                }
            )

        def connect_database(self):
            pass

    class MockFetcher:
        def fetch_all(self):
            return pd.DataFrame(
                {
                    "iso": ["NGA", "NGA"],
                    "year": [2014, 2014],
                    "type": ["current_account_balance", "gdp"],
                    "value": [-10e9, 500e9],
                }
            )

        def connect_database(self):
            pass

    engine = DataEngine(cleaner=MockCleaner(), fetcher=MockFetcher())
    engine.sync_matrix({566: "NGA"})

    matching = engine.df.query("iso == 'NGA' and year == 2014")
    assert matching.iloc[0]["current_account_balance"] == -10e9
    assert matching.iloc[0]["gdp"] == 500e9


def test_speartests_handles_missing_optional_trade_measures():
    df = pd.DataFrame(
        {
            "iso": ["GHA", "GHA"],
            "primaryvalue": [10.0, 20.0],
            "exchange_rate": [1.0, 2.0],
            "inflation": [5.0, 6.0],
        }
    )

    results = DataEngine(cleaner=None, fetcher=None)
    results.df = df

    calculated = results.speartests()

    assert calculated["Unit Value Primary Val : Net Weight (GHA): "] is None
    assert calculated["Elasticity - Quantity vs Inflation (GHA): "] is None


def test_gdp_fetch_failure_does_not_silently_continue_without_gdp(
    mock_env_variables, monkeypatch
):
    wb_request = Mock(side_effect=RuntimeError("GDP source unavailable"))
    read_fallback = Mock(side_effect=ValueError("no valid cached dataset"))
    monkeypatch.setattr("datacleanse.wb.data.DataFrame", wb_request)
    monkeypatch.setattr("datacleanse.pd.read_sql_table", read_fallback)
    fetcher = Fetcher(db_path=mock_env_variables["db_path"])

    with pytest.raises(RuntimeError, match="GDP source unavailable"):
        fetcher.fetch_all()
    assert not fetcher.is_from_fallback
    read_fallback.assert_called_once()


def test_run_swat_handles_model_analysis_value_error(monkeypatch, capsys):
    engine = Mock()
    engine.df = pd.DataFrame({"iso": ["GHA"]})
    monkeypatch.setattr(
        "main.trilateral_analysis", lambda: (engine, Mock(), Mock())
    )

    from main import run_swat

    run_swat()

    assert "SWAT Fatal: Application dashboard execution halted due to model analysis failures" in capsys.readouterr().out


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
