# SWAT EC — System Architecture

> **Status convention:** This document separates current implementation from proposed architecture. A component marked planned/recommended is not a claim that it already exists.

## 1. Purpose and scope

SWAT EC is an exploratory data-engineering and analytical system examining China-related trade in selected energy products and macroeconomic conditions in Ghana and Nigeria.

Current trade scope:
- Reporters: Ghana (GHA) and Nigeria (NGA).
- Partner: China (CHN; Comtrade partner code 156).
- Products: solar photovoltaic modules/panels (HS 854143) and electrical energy (HS 271600).
- Period configured in ingestion: 2014–2024.
- Macro indicators handled include GDP, inflation, exchange rate and current-account balance, subject to source availability and mapping validation.

Keep five information classes distinct: **observed data**, **derived measures**, **statistical estimates**, **scenario assumptions**, and **presentation outputs**.

## 2. Current repository architecture

The current repository is a Python-led pipeline orchestrated by main.py.

~~~mermaid
flowchart TD
    A[UN Comtrade API] --> B[DataCleaner]
    C[World Bank / DBnomics] --> D[Fetcher]
    B --> E[DataEngine.sync_matrix]
    D --> E
    E <--> F[(Configured database persistence)]
    E --> G[Descriptive statistics]
    E --> H[Correlations / Spearman / exploratory OLS]
    E --> I[TradeExpScore]
    I --> J[TES by country, year and energy type]
    I --> K[PCA exploratory output]
    K --> L[ECModels regression]
    J --> M[main.py]
    L --> M
    M --> N[Console output]
    O[Dashboard / report] -. planned .-> M
    P[Game theory / scenarios] -. not implemented .-> M
~~~

### Existing module responsibilities

| Module | Current responsibility | Caveat |
|---|---|---|
| config.py | Series mappings, country codes, partner code, periods/product codes and DB names | Mapping needs reconciliation with indicators actually fetched in datacleanse.py. |
| datacleanse.py | DataCleaner and Fetcher; API extraction, cleaning/standardisation, persistence and fallback | Verify source IDs, units, definitions, reporting status and safe repeat writes. |
| engine.py — DataEngine | Synchronises trade and macro data; descriptive statistics, correlations and exploratory tests | run_stats() calls meta_clean() but then copies self.df, so the cleaned return value is not used there. |
| engine.py — TradeExpScore | Aggregates trade by country/year/product/direction, joins GDP and calculates TES; also PCA | Missing directions are currently filled with zero; reporting evidence is needed. PCA is separate from TES. |
| models.py — ECModels | Uses PCA scores in a linear regression; contains forecasting and game-theory placeholders | Forecasting and game theory are pass placeholders. Regression fits and scores on the same data; it is not a validated forecast. |
| main.py | Loads environment variables, constructs dependencies and orchestrates the run | Orchestration and console presentation are combined; no separate application/API boundary is shown. |
| test.py | Existing test/diagnostic code | Confirm automated deterministic coverage and CI execution before claiming test coverage. |

## 3. Recommended target architecture

Adopt a **modular monolith** first, not premature microservices. Keep responsibilities separate while allowing the system to run as one Python application.

~~~mermaid
flowchart TB
    subgraph Sources[External data sources]
        UC[UN Comtrade]
        WB[World Bank]
        DN[DBnomics / IMF series]
    end
    subgraph Ingestion[1. Ingestion adapters]
        UCA[Comtrade adapter]
        WBA[World Bank adapter]
        DNA[DBnomics adapter]
        CFG[Central config and series registry]
    end
    subgraph Quality[2. Validation]
        SCHEMA[Schema and type validation]
        PROV[Provenance, units, frequency, reporting status]
        QA[Duplicate, null, range and key checks]
    end
    subgraph Storage[3. Persistence]
        RAW[(Raw source snapshots)]
        STG[(Staging tables)]
        CUR[(Curated analytical tables)]
    end
    subgraph Transform[4. Transformations]
        JOIN[Country-period synchronisation]
        AGG[Trade aggregation by product and direction]
        TES[TES calculation]
        FEATURES[Feature and outcome construction]
    end
    subgraph Analysis[5. Analysis]
        DESC[Descriptive statistics]
        ASSOC[Exploratory association / econometrics]
        PCA[PCA / optional]
        PRED[Prediction if sample supports it]
        GAME[Game-theory / policy scenarios]
        SENS[Sensitivity and uncertainty]
    end
    subgraph Delivery[6. Delivery]
        SERVICE[Analysis service / application interface]
        DASH[Dashboard and charts]
        REPORT[Research report / reproducible outputs]
    end
    UC --> UCA
    WB --> WBA
    DN --> DNA
    CFG --> UCA
    CFG --> WBA
    CFG --> DNA
    UCA --> SCHEMA
    WBA --> SCHEMA
    DNA --> SCHEMA
    SCHEMA --> PROV --> QA
    QA --> RAW --> STG --> CUR
    CUR --> JOIN --> AGG --> TES --> FEATURES
    FEATURES --> DESC
    FEATURES --> ASSOC
    FEATURES --> PCA
    FEATURES --> PRED
    FEATURES --> GAME
    ASSOC --> SENS
    PCA --> SENS
    PRED --> SENS
    GAME --> SENS
    DESC --> SERVICE
    ASSOC --> SERVICE
    PCA --> SERVICE
    PRED --> SERVICE
    GAME --> SERVICE
    SENS --> SERVICE
    SERVICE --> DASH
    SERVICE --> REPORT
~~~

This is a target design, not a description of the current implementation.

## 4. Data contracts and grain

Define the grain of each dataset before joining or modelling.

### 4.1 Source trade observations

**Grain:** one source-reported observation at its native dimensions (reporter, partner, product, flow, period, quantity, unit and value as available).

Retain source metadata required to interpret and reproduce each observation. Do not remove provenance before validation.

### 4.2 Macro observations

**Grain:** one indicator observation per country, period and source/series definition.

Before pivoting indicator types into columns:
- Confirm country-period-indicator keys are unique, or define a defensible aggregation/revision policy.
- Do not use aggfunc='first' to hide conflicting values.
- Retain source series ID, unit, frequency, observation status and retrieval timestamp where possible.
- Reconcile config.py with indicators actually requested by the fetcher.

### 4.3 Curated trade panel

**Grain:** country × year × energy product × trade direction after aggregation.

Keep imports and exports distinct. Preserve reporting-status evidence so an absent direction is not automatically interpreted as zero.

### 4.4 TES output

**Grain:** country × year × energy product.

Current constructed measure:

~~~text
TES = 100 × (bilateral imports from China + bilateral exports to China) / GDP
~~~

Interpretation: **bilateral energy trade intensity relative to GDP**. It is not by itself energy dependency, independence, domestic energy access, affordability, distribution or equity.

Safeguards:
- Trade values and GDP must use comparable currency units and compatible nominal/real definitions.
- Report imports, exports, net imports and combined gross trade alongside TES.
- Distinguish observed zero trade from unreported/unavailable trade.
- Validate one GDP value per country-year and a positive denominator.
- Document that the selected HS codes do not represent the entirety of national energy systems.

## 5. Analytical boundaries

### Descriptive and exploratory analysis
Summaries, distributions, correlations and rank-based associations describe the sample; they do not establish causality.

### Composite indicators
Keep TES construction separate from PCA. Compare transparent alternatives (equal weights or theory-led weights) before data-driven weights. Document orientation, normalisation, missing-data rules and sensitivity.

### PCA
PCA is optional exploratory dimensionality reduction. It maximises variance captured in input features; it does not automatically produce economically meaningful weights or a valid TES. If used predictively, fit preprocessing on training data only.

### Econometrics and prediction
Define outcome, predictors, observational unit and sample size before selecting a model. With a short annual series, prefer parsimonious analysis and explicit uncertainty. A fitted regression is not automatically a forecast. Use time-aware evaluation only where the sample supports it; otherwise label results exploratory or illustrative.

### Game theory and scenarios
Keep strategic models separate from empirical TES and regression. Define players, strategies, outcomes and player-specific utilities. Label assumed payoffs hypothetical unless calibrated from evidence. Use sensitivity analysis to show how conclusions change with assumptions.

## 6. Recommended Python package structure

This is a target structure, not the current repository layout.

~~~text
swat/
├── config/
│   ├── settings.py
│   └── series_registry.py
├── ingestion/
│   ├── comtrade.py
│   ├── world_bank.py
│   └── dbnomics.py
├── validation/
│   ├── schemas.py
│   ├── quality_checks.py
│   └── provenance.py
├── storage/
│   ├── database.py
│   └── repositories.py
├── transforms/
│   ├── trade_panel.py
│   ├── macro_panel.py
│   └── tes.py
├── analysis/
│   ├── descriptive.py
│   ├── association.py
│   ├── pca.py
│   └── diagnostics.py
├── models/
│   ├── regression.py
│   ├── forecasting.py
│   └── game_theory.py
├── scenarios/
│   ├── assumptions.py
│   └── sensitivity.py
├── application/
│   └── services.py
├── presentation/
│   ├── dashboard.py
│   └── charts.py
└── cli.py

tests/
├── test_ingestion.py
├── test_validation.py
├── test_transforms.py
├── test_tes.py
├── test_models.py
└── fixtures/
~~~

Do not split every function into a file for its own sake. Extract a module when it has a coherent responsibility, stable interface and testable contract.

## 7. Execution sequence

1. Load configuration and environment secrets.
2. Instantiate source adapters and storage dependencies.
3. Extract source observations and record retrieval metadata.
4. Validate schemas, keys, types, units, frequency and reporting status.
5. Persist source/staging data safely; make repeated runs idempotent where practical.
6. Build curated trade and macro tables at documented grains.
7. Join on validated country-period keys; report join coverage and unmatched observations.
8. Aggregate trade by product and direction.
9. Calculate TES only after GDP and reporting-status checks pass.
10. Run descriptive analysis and optional exploratory methods.
11. Run regression, forecasting or game-theory scenarios only when inputs and assumptions are defensible.
12. Export reproducible tables, figures, diagnostics and limitations to the presentation layer.

## 8. Cross-cutting engineering requirements

- **Configuration:** one authoritative registry for source IDs, countries, products, periods, units and database settings.
- **Secrets:** environment variables; never commit API keys or credentials.
- **Provenance:** source, series, retrieval date, native frequency, unit and transformation history.
- **Validation:** checks for duplicate keys, missingness, impossible values, non-positive denominators and join coverage.
- **Idempotency:** repeated extraction must not silently multiply records or overwrite source history without a documented policy.
- **Testing:** unit tests for transformations/TES; integration smoke tests for extraction-to-curated-table flow; fixtures for missing, zero and duplicate cases.
- **Observability:** logs, row counts, rejected-record counts and clear failure messages.
- **Reproducibility:** pinned dependencies, documented setup, deterministic transformations and saved model assumptions.
- **Separation of concerns:** ingestion does not calculate economic indicators; transformations do not fetch APIs; models do not own database access; presentation does not silently recalculate research measures.

## 9. Deployment boundary

**Current:** main.py runs the pipeline and prints results to the console.

**Potential next stage:** a dashboard/report consumes validated outputs through a small application/service layer. Streamlit, Plotly, D3.js or a static website can be selected at the presentation boundary; the data pipeline should not depend on one visualisation framework.

Cloud deployment, background workers, caching, queues and a public API are optional scaling choices, not prerequisites for a credible undergraduate research prototype.

## 10. Known implementation gaps

1. Reconcile config.py series mappings with actual API fetches and standardised indicator names.
2. Replace or justify aggfunc='first' after inspecting duplicate macro keys.
3. Preserve missing-versus-zero reporting status before filling absent import/export directions.
4. Correct run_stats() to use the cleaned dataframe returned by meta_clean(), or remove the unused call.
5. Add a smoke test from extraction through synchronisation, aggregation and TES; assert row counts, join coverage and usable GDP/macro observations.
6. Review PCA row filtering: the target is currently required for valid rows although it is not used to fit PCA.
7. Prevent preprocessing leakage if PCA enters a train/test workflow.
8. Add defensible evaluation before describing current in-sample regression as predictive.
9. Implement forecasting and game theory only when specifications and data/assumptions are ready; they are currently placeholders.
10. Confirm actual database engine, tables, constraints and persistence semantics before documenting a more specific storage topology.

## 11. Architecture decision

**Adopt a modular-monolith architecture with explicit data contracts and a reproducible analytical pipeline.** Prioritise trustworthy observations, transparent measures and testable transformations over infrastructure. The immediate engineering milestone is a validated extraction → staging → curated panel → TES smoke test, not microservices or complex ML deployment.
