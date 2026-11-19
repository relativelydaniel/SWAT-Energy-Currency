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
            for required_col in ['exchange_rate', 'inflation', 'hfce']:
                if required_col not in db_pivot.columns:
                    db_pivot[required_col] = np.nan

            # Forward/backward fill exchange_rate and inflation per ISO
            predictor_cols = ['exchange_rate', 'inflation']
            for col in predictor_cols:
                db_pivot[col] = db_pivot.groupby('iso')[col].transform(lambda g: g.ffill().bfill())

            # Masks for Nigeria HFCE OLS Imputation
            nga_mask = db_pivot['iso'] == 'NGA'
            train_mask = nga_mask & db_pivot['hfce'].notna() # 2014-2021
            pred_mask = nga_mask & db_pivot['hfce'].isna()   # 2022-2024
            
            features = ['exchange_rate', 'inflation']

            # Training and Imputing
            if train_mask.sum() > 0 and pred_mask.sum() > 0:
                X_train = db_pivot.loc[train_mask, features]
                y_train = db_pivot.loc[train_mask, 'hfce']
                X_pred = db_pivot.loc[pred_mask, features]

                # Fit OLS Regression
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Impute predicted HFCE values directly back into db_pivot for 2022-2024
                db_pivot.loc[pred_mask, 'hfce'] = model.predict(X_pred)
                
                print("Successfully imputed missing 2022-2024 HFCE data using OLS.")
                
            merged_df = pd.merge(uncom_df, db_pivot, on=['year', 'iso'], how='inner')
            if 'hfce' in merged_df.columns: 
                merged_df['hfce'] = merged_df['hfce'].fillna(merged_df['hfce'].mean())
            
            if merged_df.empty:
                raise RuntimeError("Data integrity failure: Inner join yielded 0 rows.")

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
        macro_cols = ['inflation', 'exchange_rate', 'hfce']
        target_cols = [col for col in macro_cols if col in df.columns]
        
        if energy_cols:
            print("\n=== UNCOM TRADE CORRELATION ===")
            print(df[energy_cols].apply(pd.to_numeric, errors='coerce').corr())
            
        if target_cols:
            print("\n=== DBN MACROECONOMIC CORRELATION ===")
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
        
        if 'altqty' in self.df.columns and 'qty' in self.df.columns:
            self.df['altqty'] = self.df['altqty'].replace(0, np.nan)
            self.df['qty_ratio'] = self.df['qty'] / self.df['altqty'] # unit based calculation for elasticicity
        
        grouped = self.df.groupby('iso')
        self.df['altqty'] = self.df['altqty'].replace(0, float('nan'))
        self.df['unit_value'] = (self.df['primaryvalue'] / self.df['netwgt'].replace(0, np.nan))    
        results = {} 
        print("\n--- SPEARMAN CORRELATION & VARIATION TESTS---")

        # highlight the ISO / country code per iteration and derive quantity \
            
        for iso, subset in grouped:   
            print(f"\n--- COUNTRY: {iso} ---")
                                                                                                                                                
            val = subset['primaryvalue'].corr(subset['exchange_rate'], method='spearman')
            print(f"Spearman - Primary Value vs Exchange Rate ({iso}): {val:.4f}")
            results[f'Spearman - Primary Value vs Exchange Rate ({iso}): '] = round(val, 4)
        
            # unit value 
            net_wgt = subset['netwgt'] 
            pv = subset['primaryvalue'] # derived value
            if pv.dropna().empty or net_wgt.sum() == 0:
                results[f'Unit Value Primary Val : Net Weight ({iso}): '] = None
                print(f"Warning: Data for {iso} is insufficient for UV calculation.")
            else:
                unit_val = subset['unit_value'].replace(0, np.nan)
                print(f"Unit Value Primary Val : Net Weight ({iso}): {unit_val.mean():.4f}")
                results[f'Unit Value Primary Val : Net Weight ({iso}): '] = round(unit_val.mean(), 4)

            
            # elasticity calculation | hybrid log log regression + division method
            inflation = subset['inflation'].replace(0, np.nan)
            qty_ratio = subset['qty_ratio'].replace(0, np.nan)
            
            valid_data = pd.concat([qty_ratio, inflation], axis=1).dropna()
            if inflation.empty or valid_data.empty or inflation.sum() == 0:
                results[f'Elasticity - Quantity vs Inflation ({iso}): '] = None
                print(f"Warning: Data for {iso} is insufficient for elasticity calculation.")
            
            if len(valid_data) >= 3:
                log_inf = np.log(valid_data['inflation'])
                log_qty = np.log(valid_data['qty_ratio'])
                X = sm.add_constant(log_inf)
                model = sm.OLS(log_qty, X).fit()
                
                elast = model.params['inflation']
                residuals = model.resid # residuals are like the standard deviation relative to the elasticity relationship
                print(f"Elasticity - Quantity vs Inflation ({iso}): {elast:.4f}")
                print(f"Residuals: {residuals.describe()}")
                results[f'Elasticity - Quantity vs Inflation ({iso}): '] = round(elast, 4)
                results[f'Residuals - Quantity vs Inflation ({iso}): '] = residuals.tolist()
            
            # Covariance
            pass
            
            
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
                    
    
    def run_pca(self, n_components=2, target_col='hfce'):
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
        
        # Principal Component Analysis based on briding EES gap
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
        gdp["gdp"] = pd.to_numeric(gdp["gdp"], errors="coerce")
        if gdp[["iso", "year", "gdp"]].isna().any().any():
            raise ValueError(
                "GDP data contains missing or non-numeric country, year, or GDP values."
            )
        if (gdp["gdp"] <= 0).any():
            raise ValueError("GDP must be greater than zero to calculate TES.")
        if gdp.duplicated(["iso", "year"]).any():
            raise ValueError("GDP data must contain one value per country and year.")

        result = totals.merge(
            gdp, on=["iso", "year"], how="left", validate="many_to_one"
        )
        if result["gdp"].isna().any():
            missing = result.loc[result["gdp"].isna(), ["iso", "year"]].drop_duplicates()
            raise ValueError(
                "GDP is missing for selected trade observations: "
                f"{missing.to_dict(orient='records')}"
            )
        result["tes"] = 100 * result["total_trade"] / result["gdp"]
        return result.sort_values(
            ["iso", "energy_type", "year"]
        ).reset_index(drop=True)
    
    
    def parse_sr(self, sr_expression):
        # Convert the symbolic regression like expression to a string based sympy expression
        raw_str = str(sr_expression) 
    
        # Map gplearn string operators to SymPy mathematical operations
        local_dict = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b,
            'log': sp.log, # respectively limits till here
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'neg': lambda a: -a,
            'inv': lambda a: 1 / a
        }
        
        # Map indexed variables (X0, X1, ...) to actual dataframe column names
        for i, name in enumerate(self.feature_names):
            local_dict[f'X{i}'] = sp.Symbol(name)
            
        try:
            sympy_expr = eval(raw_str, {"__builtins__": None}, local_dict)
            return sympy_expr
        except Exception as e:
            print(f"Error parsing symbolic regression expression: {e}")
            return None
    
    def json_dc(self): 
        return { 
            } # placeholders