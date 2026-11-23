import pandas as pd
import numpy as np
from datacleanse import DataCleaner, Fetcher
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sympy as sp
import json
import warnings
from decimal import Decimal, InvalidOperation
warnings.filterwarnings('ignore')


# DataEngine is a composition class for orchestrating the math
class DataEngine:
    def __init__(self, cleaner, fetcher):
        self.cleaner = cleaner
        self.fetcher = fetcher
        self.df = pd.DataFrame()
        
        self.metadata_cols = [
            'refperiodid', 'refmonth', 'partnercode', 'partner2code', 
            'motcode', 'qtyunitcode', 'altqtyunitcode', 'legacyestimationflag',
            'year', 'refyear', 'reportercode', 'period', 'date', 
            'isqtyestimated', 'isaltqtyestimated', 'isnetwgtestimated', 
            'isgrosswgtestimated', 'isreported', 'isaggregate'
        ]
    
        
    def sync_matrix(self, countries):
        try:
            uncom = self.cleaner.fetch_api(countries)
            dbnomics = self.fetcher.fetch_all()
            
            self.cleaner.connect_database()
            self.fetcher.connect_database()
        
            if uncom is None or uncom.empty or dbnomics is None or dbnomics.empty:
                raise ValueError("Upstream extraction returned empty datasets.")

            uncom_df = uncom.copy()
            uncom_df['year'] = uncom_df['refyear'].astype(int)
            
            dbnomics_df = dbnomics.copy()
            dbnomics_df['year'] = dbnomics_df['year'].astype(int)
            
            # In engine.py inside sync_matrix():

            db_pivot = dbnomics_df.pivot_table(
                index=['year', 'iso'],
                columns='type',
                values='value',
                aggfunc='first'
            ).reset_index()

            # GUARD: Guarantee required macro columns exist in db_pivot
            for required_col in [
                'exchange_rate',
                'inflation',
                'current_account_balance',
                'gdp',
            ]:
                if required_col not in db_pivot.columns:
                    db_pivot[required_col] = np.nan
            if db_pivot["gdp"].isna().all():
                raise ValueError("Required GDP series has no usable values after pivot.")

            if 'iso' not in uncom_df.columns:
                raise ValueError("UN Comtrade data is missing the required 'iso' column.")

            uncom_df['iso'] = uncom_df['iso'].astype(str).str.upper().str.strip()
            db_pivot['iso'] = db_pivot['iso'].astype(str).str.upper().str.strip()

            merged_df = uncom_df.merge(
                db_pivot,
                on=['year', 'iso'],
                how='left',
                validate='many_to_one'
            )

            if merged_df.empty:
                raise ValueError("No trade and macroeconomic rows matched after synchronization.")

            self.df = merged_df
            print(f"-> Matrix synchronized successfully! Matrix shape: {self.df.shape}")
        except Exception as e:
            raise RuntimeError(f"Data synchronization failed: {e}")
        
    def meta_clean(self): 
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        df_cleaned = self.df.drop(columns=self.metadata_cols, errors='ignore').copy()
        df_cleaned = df_cleaned.dropna(axis=1, how='all')
        
        non_zero_cols = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)].columns
        df_cleaned = df_cleaned[non_zero_cols]
        
        return df_cleaned  # Return the cleaned subset instead of overwriting self.df

    def run_stats(self):
        print("\nRunning full analysis:")
        if self.df.empty:
            print("Error: No matrix data present inside the engine to analyse.")
            return None
        
        print("\n--- MERGED DATA ft. First 20 rows ---")
        print(f"Dimensions: {self.df.shape}")
        print(self.df.head(20))
        
        self.meta_clean()
        
        df = self.df.copy()
        metadata = ['typecode', 'freqcode', 'iso']        
        df = df.drop(columns=metadata, errors='ignore')
        df = df.select_dtypes(include=[np.number])
        

        print("\n--- Data Types ---")
        print(self.df.dtypes)
        print(self.df.info())
        stats_summary = df.describe()
        stats_summary.loc['var'] = df.var(numeric_only=True)
        stats_summary.loc['skew'] = df.skew(numeric_only=True)
        print(stats_summary)
        return stats_summary   # individual matrix then combined matrix
    
    def run_corr(self):
        df = self.meta_clean()
        if df.empty:
            return None
       
        energy_cols = [c for c in df.columns if any(x in c for x in ['primaryvalue', 'qty', 'fobvalue', 'netwgt', 'cifvalue'])]
        macro_cols = ['inflation', 'exchange_rate', 'current_account_balance']
        target_cols = [col for col in macro_cols if col in df.columns]
        
        if energy_cols:
            print("\n=== UNCOM TRADE CORRELATION ===")
            print(df[energy_cols].apply(pd.to_numeric, errors='coerce').corr())
            
        if target_cols:
            print("\n=== WB-DBN MACROECONOMIC CORRELATION ===")
            print(df[target_cols].apply(pd.to_numeric, errors='coerce').corr())

        if target_cols and energy_cols:
            print("\n=== COMBINED CORRELATION MATRIX ===")
            combined_cols = list(set(energy_cols + target_cols))
            
            corr_df = df[combined_cols].apply(pd.to_numeric, errors='coerce')
            corr_matrix = corr_df.corr()
            print(corr_matrix)
            results = self.speartests()
            return corr_matrix, results
        return None
       
    def speartests(self):
        if self.df is None or self.df.empty:
            return {}

        data = self.df.copy()
        if {"altqty", "qty"}.issubset(data.columns):
            altqty = pd.to_numeric(data["altqty"], errors="coerce").replace(0, np.nan)
            qty = pd.to_numeric(data["qty"], errors="coerce")
            data["qty_ratio"] = qty / altqty
        if {"primaryvalue", "netwgt"}.issubset(data.columns):
            primaryvalue = pd.to_numeric(data["primaryvalue"], errors="coerce")
            netwgt = pd.to_numeric(data["netwgt"], errors="coerce").replace(
                0, np.nan
            )
            data["unit_value"] = primaryvalue / netwgt

        grouped = data.groupby('iso')
        results = {}
        print("\n--- SPEARMAN CORRELATION & VARIATION TESTS---")

        for iso, subset in grouped:
            print(f"\n--- COUNTRY: {iso} ---")

            if {"primaryvalue", "exchange_rate"}.issubset(subset.columns):
                val = pd.to_numeric(
                    subset['primaryvalue'], errors="coerce"
                ).corr(
                    pd.to_numeric(subset['exchange_rate'], errors="coerce"),
                    method='spearman',
                )
                rounded_val = round(float(val), 4) if pd.notna(val) else None
                results[
                    f'Spearman - Primary Value vs Exchange Rate ({iso}): '
                ] = rounded_val
                displayed_val = (
                    f"{rounded_val:.4f}" if rounded_val is not None else "None"
                )
                print(
                    f"Spearman - Primary Value vs Exchange Rate ({iso}): "
                    f"{displayed_val}"
                )

            if "unit_value" not in subset.columns:
                results[f'Unit Value Primary Val : Net Weight ({iso}): '] = None
                print(f"Warning: Data for {iso} is insufficient for UV calculation.")
            else:
                unit_val = subset['unit_value'].replace(0, np.nan)
                mean_unit_value = unit_val.mean()
                rounded_mean = (
                    round(float(mean_unit_value), 4)
                    if pd.notna(mean_unit_value)
                    else None
                )
                displayed_mean = (
                    f"{rounded_mean:.4f}" if rounded_mean is not None else "None"
                )
                print(
                    f"Unit Value Primary Val : Net Weight ({iso}): "
                    f"{displayed_mean}"
                )
                results[
                    f'Unit Value Primary Val : Net Weight ({iso}): '
                ] = rounded_mean

            if not {"inflation", "qty_ratio"}.issubset(subset.columns):
                results[f'Elasticity - Quantity vs Inflation ({iso}): '] = None
                print(f"Warning: Data for {iso} is insufficient for elasticity calculation.")
                continue

            valid_data = subset[["inflation", "qty_ratio"]].apply(
                pd.to_numeric, errors="coerce"
            )
            valid_data = valid_data.loc[
                (valid_data["inflation"] > 0) & (valid_data["qty_ratio"] > 0)
            ].dropna()
            if len(valid_data) < 3:
                results[f'Elasticity - Quantity vs Inflation ({iso}): '] = None
                print(f"Warning: Data for {iso} is insufficient for elasticity calculation.")
                continue

            log_inf = np.log(valid_data['inflation'])
            log_qty = np.log(valid_data['qty_ratio'])
            X = sm.add_constant(log_inf)
            model = sm.OLS(log_qty, X).fit()

            elast = model.params.iloc[-1]
            residuals = model.resid
            rounded_elast = round(float(elast), 4)
            rounded_residuals = residuals.round(4)
            print(
                f"Elasticity - Quantity vs Inflation ({iso}): "
                f"{rounded_elast:.4f}"
            )
            print(f"Residuals: {residuals.describe().round(4)}")
            results[
                f'Elasticity - Quantity vs Inflation ({iso}): '
            ] = rounded_elast
            results[f'Residuals - Quantity vs Inflation ({iso}): '] = (
                rounded_residuals.tolist()
            )
        return results
    # END OF FIRST HALF 

 
 
 # Calculations - guided with PCA + OLS
 # Trade Exposure Score Gap (economic + energy value(s) as key inds)
 
class TradeExpScore:
    def __init__(self, df):
        self.df = df
        self.features = ['netwgt', 'inflation', 'exchange_rate', 'primaryvalue'] # Store feature names for Symbolic Regression reminiscent weighting
        self.feature_names = [col for col in self.features if col in self.df.columns]
        target_col = None
                    
    
    def run_pca(self, n_components=2, target_col='current_account_balance'):
        if self.df is None or self.df.empty:
            return None
        
        active_features = [col for col in self.features if col in self.df.columns] # = feature_names
        
        if not active_features:
            print("Warning: No valid PCA features found.")
            return None

        # Require target
        if target_col not in self.df.columns:
            print(f"Warning: Target '{target_col}' missing.")
            return None
        
        req_cols = active_features + [target_col]
        valid_df = self.df.dropna(subset=req_cols).copy()

        if valid_df.empty:
            print("Warning: No valid rows available for PCA.")
            return None
        
        # PCA is exploratory; TES is calculated separately from trade and GDP.
        X = valid_df[active_features]
        
        X = valid_df[active_features].copy()
        for feature in ("netwgt", "primaryvalue"):
            if feature in X.columns:
                X[feature] = np.log1p(X[feature].clip(lower=0))        
        Y = valid_df[target_col]

        # Standardize the data for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaled = pd.DataFrame(X_scaled, columns=active_features, index=valid_df.index)
        
        # Fit on training data AND transform it
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        loadings = pd.DataFrame(
        pca.components_.T,
        index=active_features,
        columns=[f'PC{i + 1}' for i in range(n_components)])

        print("\n--- PCA LOADINGS ---")
        print(loadings)

        print("\n--- EXPLAINED VARIANCE ---")
        for i, variance in enumerate(pca.explained_variance_ratio_):
            print(f"PC{i + 1}: {variance:.4f} " f"({variance * 100:.2f}%)")
            
        pca_results = {
        'pca_model': pca,
        'scaler': scaler,
        'scaled_data': scaled,
        'pca_scores': X_pca,
        'loadings': loadings,
        'explained_variance': pca.explained_variance_ratio_,
        'feature_names': active_features }

        return pca_results, scaled
        
    
    def calculate_tes(self, gdp_df):
        """Calculate bilateral TES by country, energy type, and year.

        Trade data must include ``iso``, ``year`` (or ``refyear``),
        ``partnercode``, ``cmdcode``, ``flowcode``, and ``primaryvalue``.
        GDP data must include ``iso``, ``year``, and ``gdp`` in the same
        currency units as the trade values.
        """
        required_trade = {
            "iso", "partnercode", "cmdcode", "flowcode", "primaryvalue"
        }
        missing_trade = required_trade.difference(self.df.columns)
        if missing_trade:
            raise ValueError(
                f"Trade data is missing required columns: {sorted(missing_trade)}"
            )
        if "year" not in self.df.columns and "refyear" not in self.df.columns:
            raise ValueError("Trade data must contain either 'year' or 'refyear'.")
        required_gdp = {"iso", "year", "gdp"}
        missing_gdp = required_gdp.difference(gdp_df.columns)
        if missing_gdp:
            raise ValueError(
                f"GDP data is missing required columns: {sorted(missing_gdp)}"
            )

        trade = self.df.copy()
        if "year" not in trade.columns:
            trade = trade.rename(columns={"refyear": "year"})
        trade["year"] = pd.to_numeric(trade["year"], errors="coerce")
        trade["partnercode"] = pd.to_numeric(trade["partnercode"], errors="coerce")
        def normalize_cmdcode(value):
            raw_code = str(value).strip()
            try:
                decimal_code = Decimal(raw_code)
            except InvalidOperation:
                return raw_code
            if not decimal_code.is_finite() or decimal_code != decimal_code.to_integral_value():
                return raw_code
            return str(int(decimal_code))

        trade["cmdcode"] = trade["cmdcode"].map(normalize_cmdcode)
        trade["flowcode"] = trade["flowcode"].astype(str).str.upper().str.strip()
        trade = trade.loc[trade["flowcode"].isin({"M", "X"})].copy()
        trade["primaryvalue"] = pd.to_numeric(
            trade["primaryvalue"], errors="coerce"
        )
        if trade["partnercode"].isna().any():
            raise ValueError("Trade data contains rows with an unknown partner code.")

        product_codes = {"854143": "solar_pv", "271600": "electricity"}
        trade = trade.loc[
            trade["partnercode"].eq(156)
            & trade["cmdcode"].isin(product_codes)
        ].copy()
        if trade.empty:
            raise ValueError(
                "No China-partner trade rows match the configured energy products."
            )
        if trade[["iso", "year", "flowcode", "primaryvalue"]].isna().any().any():
            raise ValueError(
                "Selected trade rows contain missing country, year, flow, or value."
            )
        if (trade["primaryvalue"] < 0).any():
            raise ValueError("Trade values must be non-negative to calculate TES.")

        invalid_flows = set(trade["flowcode"]) - {"M", "X"}
        if invalid_flows:
            raise ValueError(
                f"Unsupported Comtrade flow codes: {sorted(invalid_flows)}; "
                "expected 'M'/'X'."
            )

        trade["energy_type"] = trade["cmdcode"].map(product_codes)
        trade["direction"] = trade["flowcode"].map(
            {"M": "imports", "X": "exports"}
        )
        totals = (
            trade.groupby(
                ["iso", "year", "energy_type", "direction"], as_index=False
            )["primaryvalue"]
            .sum()
            .pivot(
                index=["iso", "year", "energy_type"],
                columns="direction",
                values="primaryvalue",
            )
            .reset_index()
        )
        totals.columns.name = None
        for direction in ("imports", "exports"):
            if direction not in totals.columns:
                totals[direction] = 0.0
            totals[direction] = totals[direction].fillna(0.0)
        totals["total_trade"] = totals["imports"] + totals["exports"]

        gdp = gdp_df[["iso", "year", "gdp"]].copy()
        gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce")
        raw_gdp = gdp["gdp"]
        gdp["gdp"] = pd.to_numeric(gdp["gdp"], errors="coerce")
        if gdp[["iso", "year"]].isna().any().any():
            raise ValueError(
                "GDP data contains missing or non-numeric country/year keys."
            )
        nonnumeric_gdp = raw_gdp.notna() & gdp["gdp"].isna()
        if nonnumeric_gdp.any():
            raise ValueError("GDP data contains non-numeric GDP values.")
        if (gdp["gdp"] <= 0).any():
            raise ValueError("GDP must be greater than zero to calculate TES.")
        if gdp.duplicated(["iso", "year"]).any():
            raise ValueError("GDP data must contain one value per country and year.")

        result = totals.merge(
            gdp, on=["iso", "year"], how="left", validate="many_to_one"
        )
        if result["gdp"].isna().any():
            missing = result.loc[
                result["gdp"].isna(), ["iso", "energy_type", "year"]
            ].to_dict(orient="records")
            raise ValueError(
                f"GDP is required to calculate TES; missing observations: {missing}"
            )
            
        result["tes"] = 100 * result["total_trade"] / result["gdp"]
        return result.sort_values(
            ["iso", "energy_type", "year"]
        ).reset_index(drop=True)
        
    
    def json_dc(self): 
        return { 
                
            } # placeholders