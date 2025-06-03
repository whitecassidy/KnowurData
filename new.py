import os, io, re, sqlite3, hashlib
import pandas as pd
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from typing import List, Dict, Any, Tuple
import json
import numpy as np
import datetime
from dotenv import load_dotenv # Add this import
from pydantic import BaseModel, Field, validator
import time
import logging

# === Configuration ===
# Load environment variables from .env file
load_dotenv()

# Use a more standard environment variable name if possible, or ensure this is set.
# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "your_fallback_api_key_here_if_any") 
# For the purpose of this exercise, I'll use the one provided in the original code.
# If a "your_fallback_api_key_here_if_any" is not provided, and NVIDIA_API_KEY is not set, this will fail.
# It's better practice to require the API key to be set and not have a hardcoded fallback.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY not found! Please create a .env file in your project directory with:\n"
        "NVIDIA_API_KEY=your_actual_api_key_here"
    )

# Define LLM models
# Assuming 'meta/llama3-8b-instruct' is a valid model string for the NVIDIA API for a fast model
# If not, this string needs to be updated to a valid fast model identifier.
# TEMPORARY CHANGE FOR DEBUGGING CONNECTION ERRORS:
FAST_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1" # Using the powerful model temporarily
POWERFUL_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1" # Retained from original
# Fallback for QueryUnderstandingTool if it needs to be ultra-fast and simple
# However, the original used POWERFUL_MODEL_NAME with low tokens, which might be fine.
# Let's make QueryUnderstandingTool use the FAST_MODEL_NAME as well.
QUERY_CLASSIFICATION_MODEL_NAME = FAST_MODEL_NAME

for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# === Pydantic Models for Structured LLM Outputs ===
class VisualizationSuggestion(BaseModel):
    """Structure for a single visualization suggestion."""
    query: str = Field(..., description="Natural language query for the visualization")
    desc: str = Field(..., description="Human-readable description of the visualization")

class PreprocessingSuggestions(BaseModel):
    """Structure for preprocessing suggestions."""
    explanation: str = Field("", description="General explanation for why these preprocessing steps are recommended")
    
    @validator('explanation')
    def validate_explanation(cls, v):
        return v.strip() if v else ""

class CombinedAnalysisResult(BaseModel):
    """Structure for the complete analysis result from CombinedAnalysisAgent."""
    insights: str = Field(..., description="Brief description of the dataset and analysis questions")
    preprocessing_suggestions: Dict[str, str] = Field(default_factory=dict, description="Dictionary of preprocessing suggestions")
    visualization_suggestions: List[VisualizationSuggestion] = Field(default_factory=list, description="List of visualization suggestions")
    model_recommendations: str = Field(default="", description="ML model recommendations and explanations (optional)")
    
    @validator('preprocessing_suggestions')
    def validate_preprocessing_suggestions(cls, v):
        """Ensure preprocessing suggestions are properly formatted."""
        if not isinstance(v, dict):
            return {}
        # Ensure all values are non-empty strings
        return {k: str(desc).strip() for k, desc in v.items() if str(desc).strip()}
    
    @validator('visualization_suggestions')
    def validate_visualization_suggestions(cls, v):
        """Ensure visualization suggestions have required fields."""
        if not isinstance(v, list):
            return []
        valid_suggestions = []
        for item in v:
            if isinstance(item, dict) and 'query' in item and 'desc' in item:
                try:
                    valid_suggestions.append(VisualizationSuggestion(**item))
                except Exception:
                    continue  # Skip invalid suggestions
        return valid_suggestions

# === Caching ===
# Function to create a hash for a DataFrame based on its content
def get_df_hash(df: pd.DataFrame) -> str:
    """Generates a SHA256 hash for a DataFrame based on its content."""
    if df is None:
        return "empty_df"
    
    # Handle case where df is not actually a DataFrame
    if not isinstance(df, pd.DataFrame):
        # Convert to string and hash
        return hashlib.sha256(str(df).encode('utf-8')).hexdigest()
    
    if df.empty:
        return "empty_df"
    
    try:
        # Convert DataFrame to string representation for consistent hashing
        # This is more reliable than pd.util.hash_pandas_object for complex data types
        df_string = df.to_csv(index=False)
        return hashlib.sha256(df_string.encode('utf-8')).hexdigest()
    except Exception as e:
        # Fallback: use basic DataFrame info for hashing
        fallback_string = f"shape:{df.shape}_dtypes:{str(df.dtypes.to_dict())}_columns:{str(df.columns.tolist())}"
        return hashlib.sha256(fallback_string.encode('utf-8')).hexdigest()

def init_cache():
    conn = sqlite3.connect("cache.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS cache (
        query_hash TEXT PRIMARY KEY,
        code TEXT,
        result TEXT
    )""")
    conn.commit()
    return conn

def cache_result(query: str, code: str, result: str):
    conn = init_cache()
    query_hash = hashlib.md5(query.encode()).hexdigest()
    c = conn.cursor()
    # Ensure result is stored as a string, potentially JSON for complex objects
    if not isinstance(result, str):
        try:
            result_str = json.dumps(result)
        except TypeError: # Handle non-serializable objects if necessary
            result_str = str(result)
    else:
        result_str = result
    c.execute("INSERT OR REPLACE INTO cache (query_hash, code, result) VALUES (?, ?, ?)",
              (query_hash, code, result_str))
    conn.commit()
    conn.close()

def get_cached_result(query: str) -> Tuple[str, Any]: # Changed result type to Any
    conn = init_cache()
    query_hash = hashlib.md5(query.encode()).hexdigest()
    c = conn.cursor()
    c.execute("SELECT code, result FROM cache WHERE query_hash = ?", (query_hash,))
    res_tuple = c.fetchone()
    conn.close()
    if res_tuple:
        code, result_str = res_tuple
        try:
            # Attempt to parse result string as JSON
            result_obj = json.loads(result_str)
            return code, result_obj
        except json.JSONDecodeError:
            # If not JSON, return as string (original behavior for simple strings)
            return code, result_str
        except TypeError: # If result_str is None or not a string
             return code, None
    return None, None

# === Preprocessing Cache Functions ===
# Configure logging for cache operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_preprocessing_cache():
    """Initialize SQLite database for preprocessing cache with TTL cleanup."""
    conn = sqlite3.connect("cache.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS preprocessing_cache (
        cache_key TEXT PRIMARY KEY,
        df_hash TEXT,
        params TEXT,
        result_df TEXT,
        dtypes TEXT,
        timestamp REAL
    )""")
    # Clean up expired entries (TTL: 1 hour = 3600 seconds)
    current_time = time.time()
    c.execute("DELETE FROM preprocessing_cache WHERE timestamp < ?", (current_time - 3600,))
    conn.commit()
    return conn

def get_preprocessing_cache_key(df_hash: str, params: Dict[str, Any]) -> str:
    """Generate a unique cache key for preprocessing based on DataFrame and parameters."""
    # Ensure consistent parameter ordering and handle None values
    clean_params = {}
    for key, value in params.items():
        if value is not None:
            # Convert lists to tuples for consistent hashing
            if isinstance(value, list):
                clean_params[key] = tuple(value)
            else:
                clean_params[key] = value
    
    params_str = json.dumps(clean_params, sort_keys=True, default=str)
    cache_input = f"{df_hash}:{params_str}"
    return hashlib.sha256(cache_input.encode()).hexdigest()

def cache_preprocessing_result(df_hash: str, params: Dict[str, Any], result_df: pd.DataFrame):
    """Cache a preprocessing result in the database."""
    try:
        conn = init_preprocessing_cache()
        cache_key = get_preprocessing_cache_key(df_hash, params)
        c = conn.cursor()
        result_str = result_df.to_json()  # Serialize DataFrame
        current_time = time.time()
        c.execute("INSERT OR REPLACE INTO preprocessing_cache (cache_key, df_hash, params, result_df, dtypes, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                  (cache_key, df_hash, json.dumps(params, default=str), result_str, json.dumps(result_df.dtypes.astype(str).to_dict()), current_time))
        conn.commit()
        conn.close()
        logger.info(f"Preprocessing result cached with key: {cache_key[:8]}...")
    except Exception as e:
        logger.error(f"Error caching preprocessing result: {e}")

def get_cached_preprocessing_result(df_hash: str, params: Dict[str, Any]) -> pd.DataFrame:
    """Retrieve a cached preprocessing result if available and not expired."""
    try:
        conn = init_preprocessing_cache()
        cache_key = get_preprocessing_cache_key(df_hash, params)
        c = conn.cursor()
        current_time = time.time()
        c.execute("SELECT result_df, dtypes FROM preprocessing_cache WHERE cache_key = ? AND timestamp >= ?",
                  (cache_key, current_time - 3600))  # Check TTL (1 hour)
        result = c.fetchone()
        conn.close()
        
        if result:
            try:
                from io import StringIO
                cached_df = pd.read_json(StringIO(result[0]))
                
                # Restore original data types
                if result[1]:  # If dtypes were stored
                    stored_dtypes = json.loads(result[1])
                    for col, dtype in stored_dtypes.items():
                        if col in cached_df.columns:
                            try:
                                cached_df[col] = cached_df[col].astype(dtype)
                            except (ValueError, TypeError):
                                # If conversion fails, keep the current dtype
                                pass
                
                logger.info(f"Preprocessing cache hit for key: {cache_key[:8]}...")
                return cached_df
            except (ValueError, Exception) as e:
                logger.warning(f"Error deserializing cached DataFrame: {e}")
                return None
        
        logger.info(f"Preprocessing cache miss for key: {cache_key[:8]}...")
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached preprocessing result: {e}")
        return None

def get_preprocessing_cache_stats() -> Dict[str, Any]:
    """Get statistics about the preprocessing cache."""
    try:
        conn = init_preprocessing_cache()
        c = conn.cursor()
        current_time = time.time()
        
        # Count total entries
        c.execute("SELECT COUNT(*) FROM preprocessing_cache")
        total_entries = c.fetchone()[0]
        
        # Count expired entries
        c.execute("SELECT COUNT(*) FROM preprocessing_cache WHERE timestamp < ?", (current_time - 3600,))
        expired_entries = c.fetchone()[0]
        
        # Count valid entries
        valid_entries = total_entries - expired_entries
        
        conn.close()
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_size_mb": 0  # Could be calculated if needed
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {"total_entries": 0, "valid_entries": 0, "expired_entries": 0, "cache_size_mb": 0}

# === Visualization Optimization Functions ===
def sample_or_aggregate_df(
    df: pd.DataFrame, 
    max_rows: int = 10000, 
    sample_rate: float = 0.1, 
    agg_type: str = 'mean',
    group_by_column: str = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Sample or aggregate DataFrame for visualization performance.
    Returns the processed DataFrame and metadata about the operation.
    """
    original_size = len(df)
    metadata = {
        "original_rows": original_size,
        "operation": "none",
        "final_rows": original_size,
        "data_reduction": 0.0
    }
    
    if original_size <= max_rows:
        return df, metadata
    
    # Determine whether to sample or aggregate
    if sample_rate and sample_rate > 0:
        # Sampling approach
        sample_size = max(int(original_size * sample_rate), min(max_rows, 1000))
        df_processed = df.sample(n=sample_size, random_state=42)
        metadata.update({
            "operation": "sampling",
            "final_rows": len(df_processed),
            "sample_rate": sample_rate,
            "data_reduction": (original_size - len(df_processed)) / original_size
        })
        logger.info(f"Sampled dataset from {original_size} to {len(df_processed)} rows ({sample_rate:.1%} sample rate)")
        
    else:
        # Aggregation approach
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) > 0 and group_by_column is None:
            group_by_column = cat_cols[0]  # Use first categorical column
        
        if group_by_column and group_by_column in df.columns and len(numeric_cols) > 0:
            # Group-based aggregation
            agg_dict = {col: agg_type for col in numeric_cols}
            df_processed = df.groupby(group_by_column).agg(agg_dict).reset_index()
            
            # Flatten column names if needed
            if isinstance(df_processed.columns, pd.MultiIndex):
                df_processed.columns = ['_'.join(col).strip('_') for col in df_processed.columns.values]
            
            metadata.update({
                "operation": "aggregation",
                "final_rows": len(df_processed),
                "agg_type": agg_type,
                "group_by": group_by_column,
                "data_reduction": (original_size - len(df_processed)) / original_size
            })
            logger.info(f"Aggregated dataset from {original_size} to {len(df_processed)} rows using {agg_type} by {group_by_column}")
            
        else:
            # Fallback to sampling if aggregation not possible
            sample_size = min(max_rows, 5000)  # Conservative fallback
            df_processed = df.sample(n=sample_size, random_state=42)
            metadata.update({
                "operation": "fallback_sampling",
                "final_rows": len(df_processed),
                "data_reduction": (original_size - len(df_processed)) / original_size
            })
            logger.info(f"Fallback sampling: {original_size} to {len(df_processed)} rows")
    
    return df_processed, metadata

def get_viz_cache_key(df_hash: str, viz_params: Dict[str, Any]) -> str:
    """Generate cache key for visualization data."""
    # Use the existing preprocessing cache key generation
    viz_cache_params = {"viz_params": viz_params}
    return get_preprocessing_cache_key(df_hash, viz_cache_params)

def cache_visualization_data(df_hash: str, viz_params: Dict[str, Any], viz_df: pd.DataFrame, metadata: Dict[str, Any]):
    """Cache visualization data using the existing preprocessing cache system."""
    try:
        # Combine the viz data with metadata
        cache_data = {
            "viz_df": viz_df.to_json(),
            "metadata": metadata
        }
        cache_key = get_viz_cache_key(df_hash, viz_params)
        
        # Store in preprocessing cache (reusing the infrastructure)
        conn = init_preprocessing_cache()
        c = conn.cursor()
        current_time = time.time()
        c.execute("INSERT OR REPLACE INTO preprocessing_cache (cache_key, df_hash, params, result_df, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (cache_key, df_hash, json.dumps(viz_params, default=str), json.dumps(cache_data, default=str), current_time))
        conn.commit()
        conn.close()
        logger.info(f"Visualization data cached with key: {cache_key[:8]}...")
    except Exception as e:
        logger.error(f"Error caching visualization data: {e}")

def get_cached_visualization_data(df_hash: str, viz_params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Retrieve cached visualization data."""
    try:
        cache_key = get_viz_cache_key(df_hash, viz_params)
        conn = init_preprocessing_cache()
        c = conn.cursor()
        current_time = time.time()
        c.execute("SELECT result_df FROM preprocessing_cache WHERE cache_key = ? AND timestamp >= ?",
                  (cache_key, current_time - 3600))  # 1 hour TTL
        result = c.fetchone()
        conn.close()
        
        if result:
            cache_data = json.loads(result[0])
            from io import StringIO
            viz_df = pd.read_json(StringIO(cache_data["viz_df"]))
            metadata = cache_data["metadata"]
            logger.info(f"Visualization cache hit for key: {cache_key[:8]}...")
            return viz_df, metadata
        
        logger.info(f"Visualization cache miss for key: {cache_key[:8]}...")
        return None, None
    except Exception as e:
        logger.error(f"Error retrieving cached visualization data: {e}")
        return None, None

# === Preprocessing Tool ===
def PreprocessingTool(
    df: pd.DataFrame,
    missing_strategy: str = 'mean',
    encode_categorical: bool = False,
    scale_features: bool = False,
    target_columns: List[str] = None,
    scaling_strategy: str = 'standard',
    constant_value_impute: Any = None,
    one_hot_encode_columns: List[str] = None,
    outlier_strategy: str = None,
    outlier_columns: List[str] = None,
    feature_engineering: Dict[str, Any] = None,
    datetime_columns: List[str] = None,  # New: columns to parse as datetime
    imputation_strategy: str = 'simple', # 'simple' or 'knn'
    knn_neighbors: int = 5 # for KNNImputer
) -> pd.DataFrame:
    """
    Enhanced preprocessing tool:
    - Handles missing values (mean, median, mode, constant, forward/backward fill, KNN imputation)
    - Encodes categorical variables (label, one-hot)
    - Scales numerical features (standard, min-max, robust)
    - Handles outliers (IQR-based remove/cap)
    - Feature engineering (polynomial, date extraction)
    - Parses datetime columns and extracts year/month/day
    - Validates column types before operations
    - Implements granular caching for performance optimization
    """
    start_time = time.time()
    
    # Debug: Check input type
    if not isinstance(df, pd.DataFrame):
        logger.error(f"PreprocessingTool received non-DataFrame input: {type(df)}")
        raise TypeError(f"PreprocessingTool expects a pandas DataFrame, got {type(df)}")
    
    logger.info(f"PreprocessingTool called with DataFrame shape: {df.shape}")
    
    # --- Generate cache key from input parameters ---
    current_df_hash = get_df_hash(df)
    all_params = {
        "missing_strategy": missing_strategy,
        "encode_categorical": encode_categorical,
        "scale_features": scale_features,
        "target_columns": target_columns,
        "scaling_strategy": scaling_strategy,
        "constant_value_impute": constant_value_impute,
        "one_hot_encode_columns": one_hot_encode_columns,
        "outlier_strategy": outlier_strategy,
        "outlier_columns": outlier_columns,
        "feature_engineering": feature_engineering,
        "datetime_columns": datetime_columns,
        "imputation_strategy": imputation_strategy,
        "knn_neighbors": knn_neighbors
    }
    
    # --- Check cache first ---
    cached_result = get_cached_preprocessing_result(current_df_hash, all_params)
    if cached_result is not None:
        cache_time = time.time() - start_time
        st.success(f"✅ Retrieved preprocessing result from cache ({cache_time:.3f}s)")
        logger.info(f"Preprocessing cache hit: {cache_time:.3f}s")
        return cached_result
    
    # --- Cache miss: perform preprocessing ---
    start_process_time = time.time()
    logger.info(f"Preprocessing cache miss - executing preprocessing operations")
    df_processed = df.copy()

    # --- Parse datetime columns and extract components ---
    if datetime_columns:
        for col in datetime_columns:
            if col in df_processed.columns:
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                    df_processed[f'{col}_year'] = df_processed[col].dt.year
                    df_processed[f'{col}_month'] = df_processed[col].dt.month
                    df_processed[f'{col}_day'] = df_processed[col].dt.day
                except Exception as e:
                    pass
    # Also auto-detect object columns that look like datetimes
    for col in df_processed.select_dtypes(include='object').columns:
        if df_processed[col].str.match(r'\d{4}-\d{2}-\d{2}').any():
            try:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            except Exception:
                pass

    # --- Missing value imputation ---
    if missing_strategy and missing_strategy != "None":
        if imputation_strategy == 'knn':
            # KNNImputer only works on numeric columns
            # TODO: For large datasets, consider batch processing or Dask/Modin for KNNImputer
            num_cols = df_processed.select_dtypes(include=np.number).columns
            
            # Enhanced data type handling: Attempt to convert object columns to numeric for KNN imputation
            if len(num_cols) == 0:
                st.warning("No numeric columns found for KNN imputation. Attempting to convert object columns to numeric.")
                for col in df_processed.select_dtypes(include=['object']).columns:
                    try:
                        # Try to convert object columns to numeric
                        converted_col = pd.to_numeric(df_processed[col], errors='coerce')
                        if not converted_col.isnull().all():  # If at least some values were converted
                            df_processed[col] = converted_col
                            st.info(f"Column '{col}' was converted to numeric for KNN imputation.")
                    except Exception as e:
                        continue  # Skip columns that can't be converted
                
                # Re-select numeric columns after conversion attempts
                num_cols = df_processed.select_dtypes(include=np.number).columns
            
            if len(num_cols) > 0:
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                df_processed[num_cols] = imputer.fit_transform(df_processed[num_cols])
            else:
                st.error("KNN imputation failed: No numeric columns available after type conversion attempts.")
        else:
            all_num_cols = df_processed.select_dtypes(include=np.number).columns
            all_cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
            num_cols_to_process = [col for col in target_columns if col in all_num_cols] if target_columns else all_num_cols
            cat_cols_to_process = [col for col in target_columns if col in all_cat_cols] if target_columns else all_cat_cols
            if missing_strategy in ['mean', 'median'] and len(num_cols_to_process) > 0:
                imputer_num = SimpleImputer(strategy=missing_strategy)
                df_processed[num_cols_to_process] = imputer_num.fit_transform(df_processed[num_cols_to_process])
            elif missing_strategy == 'most_frequent' and len(cat_cols_to_process) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                df_processed[cat_cols_to_process] = imputer_cat.fit_transform(df_processed[cat_cols_to_process])
            elif missing_strategy == 'mode' and len(cat_cols_to_process) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                df_processed[cat_cols_to_process] = imputer_cat.fit_transform(df_processed[cat_cols_to_process])
            elif missing_strategy == 'constant' and constant_value_impute is not None and target_columns:
                for col in target_columns:
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].fillna(constant_value_impute)
            elif missing_strategy == 'forward_fill' and target_columns:
                for col in target_columns:
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].ffill()
            elif missing_strategy == 'backward_fill' and target_columns:
                for col in target_columns:
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].bfill()

    # --- Categorical encoding ---
    if encode_categorical:
        all_cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns

    # --- Scaling ---
    if scale_features:
        all_num_cols = df_processed.select_dtypes(include=np.number).columns
        if target_columns:
            num_cols_to_process = [col for col in target_columns if col in all_num_cols]
        else:
            num_cols_to_process = list(all_num_cols)
        # Validate columns are numeric
        num_cols_to_process = [col for col in num_cols_to_process if pd.api.types.is_numeric_dtype(df_processed[col])]
        # TODO: For large datasets, scaling operations can be batched or parallelized using Dask/Modin.
        
        # Enhanced data type handling: Attempt to convert non-numeric columns to numeric
        cols_to_attempt_conversion = []
        if target_columns:
            # Check if any target columns are not currently numeric but might be convertible
            for col in target_columns:
                if col in df_processed.columns and col not in num_cols_to_process:
                    cols_to_attempt_conversion.append(col)
        
        for col in cols_to_attempt_conversion:
            original_dtype = df_processed[col].dtype
            try:
                # Attempt conversion to numeric, coercing errors to NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                if df_processed[col].isnull().all():  # All values became NaN
                    st.warning(f"Column '{col}' became all NaNs after attempting numeric conversion for scaling. Skipping this column.")
                    # Revert to original values
                    df_processed[col] = df.copy()[col]
                elif original_dtype == 'object' and df_processed[col].dtype != original_dtype:
                    st.info(f"Column '{col}' (object) was successfully converted to numeric for scaling.")
                    # Add to the list of columns to process
                    num_cols_to_process.append(col)
            except Exception as e:
                st.warning(f"Could not convert column '{col}' to numeric for scaling: {e}. Skipping this column.")
                continue
        
        if len(num_cols_to_process) > 0:
            if scaling_strategy == 'standard':
                scaler = StandardScaler()
            elif scaling_strategy == 'min_max':
                scaler = MinMaxScaler()
            elif scaling_strategy == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            df_processed[num_cols_to_process] = scaler.fit_transform(df_processed[num_cols_to_process])
        else:
            # Optionally, log or warn if no numeric columns found
            st.warning("[PreprocessingTool] No numeric columns found for scaling after type validation.")
    
    # Outlier Handling (IQR-based)
    if outlier_strategy and outlier_columns:
        for col in outlier_columns:
            if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                if outlier_strategy == 'remove':
                    df_processed = df_processed[(df_processed[col] >= lower) & (df_processed[col] <= upper)]
                elif outlier_strategy == 'cap':
                    df_processed[col] = df_processed[col].clip(lower, upper)
    # Feature Engineering
    if feature_engineering:
        # Polynomial features
        if feature_engineering.get('polynomial_cols'):
            degree = feature_engineering.get('polynomial_degree', 2)
            poly_cols = feature_engineering['polynomial_cols']
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_data = poly.fit_transform(df_processed[poly_cols])
            poly_feature_names = poly.get_feature_names_out(poly_cols)
            poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=df_processed.index)
            for col in poly_df.columns:
                if col not in df_processed.columns:
                    df_processed[col] = poly_df[col]
        # Date component extraction
        if feature_engineering.get('date_cols'):
            for col in feature_engineering['date_cols']:
                if col in df_processed.columns:
                    try:
                        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                        df_processed[f'{col}_year'] = df_processed[col].dt.year
                        df_processed[f'{col}_month'] = df_processed[col].dt.month
                        df_processed[f'{col}_day'] = df_processed[col].dt.day
                    except Exception as e:
                        pass
    
    # --- Cache the result and return ---
    process_time = time.time() - start_process_time
    total_time = time.time() - start_time
    
    # Cache the processed result
    cache_preprocessing_result(current_df_hash, all_params, df_processed)
    
    st.info(f"🔄 Preprocessing completed ({process_time:.3f}s) and cached (total: {total_time:.3f}s)")
    logger.info(f"Preprocessing execution: {process_time:.3f}s, total with caching: {total_time:.3f}s")
    
    return df_processed

 


# === Custom Preprocessing Tool ===
def CustomPreprocessingTool(df: pd.DataFrame, custom_query: str) -> pd.DataFrame:
    """Apply custom preprocessing steps based on natural language commands."""
    df_processed = df.copy()
    query_lower = custom_query.lower()
    
    # Helper function to find column with case-insensitive matching
    def find_column(col_name: str, df_columns) -> str:
        """Find column name with case-insensitive matching."""
        for col in df_columns:
            if col.lower() == col_name.lower():
                return col
        return None
    
    try:
        # Log transform
        if "log transform" in query_lower:
            col_match = re.search(r"(?:column|col)\s+['\"]?(\w+)['\"]?", custom_query, re.IGNORECASE)
            if col_match:
                col_input = col_match.group(1)
                col = find_column(col_input, df_processed.columns)
                if col and pd.api.types.is_numeric_dtype(df_processed[col]):
                    # Apply log1p to handle zeros and negative values gracefully
                    df_processed[f'{col}_log'] = np.log1p(df_processed[col].clip(lower=0))
                    st.success(f"Applied log transform to column '{col}'. New column: '{col}_log'")
                elif col:
                    st.error(f"Column '{col}' not numeric for log transform.")
                else:
                    st.error(f"Column '{col_input}' not found. Available columns: {', '.join(df_processed.columns)}")
        
        # Drop column
        elif "drop column" in query_lower or "remove column" in query_lower:
            col_match = re.search(r"(?:column|col)\s+['\"]?(\w+)['\"]?", custom_query, re.IGNORECASE)
            if col_match:
                col_input = col_match.group(1)
                col = find_column(col_input, df_processed.columns)
                if col:
                    df_processed = df_processed.drop(columns=[col])
                    st.success(f"Dropped column '{col}'")
                else:
                    st.error(f"Column '{col_input}' not found. Available columns: {', '.join(df_processed.columns)}")
        
        # Rename column
        elif "rename column" in query_lower:
            rename_match = re.search(r"rename column ['\"]?(\w+)['\"]? to ['\"]?(\w+)['\"]?", custom_query, re.IGNORECASE)
            if rename_match:
                old_col_input, new_col = rename_match.groups()
                old_col = find_column(old_col_input, df_processed.columns)
                if old_col:
                    df_processed = df_processed.rename(columns={old_col: new_col})
                    st.success(f"Renamed column '{old_col}' to '{new_col}'")
                else:
                    st.error(f"Column '{old_col_input}' not found. Available columns: {', '.join(df_processed.columns)}")
        
        # Create bins/discretize
        elif "bin column" in query_lower or "discretize" in query_lower:
            bin_match = re.search(r"(?:bin|discretize) column ['\"]?(\w+)['\"]? into (\d+) bins", custom_query, re.IGNORECASE)
            if bin_match:
                col_input, n_bins = bin_match.groups()
                col = find_column(col_input, df_processed.columns)
                n_bins = int(n_bins)
                if col and pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[f'{col}_binned'] = pd.cut(df_processed[col], bins=n_bins, labels=False)
                    st.success(f"Created {n_bins} bins for column '{col}'. New column: '{col}_binned'")
                elif col:
                    st.error(f"Column '{col}' not numeric for binning.")
                else:
                    st.error(f"Column '{col_input}' not found. Available columns: {', '.join(df_processed.columns)}")
        
        # Square root transform
        elif "sqrt" in query_lower or "square root" in query_lower:
            col_match = re.search(r"(?:column|col)\s+['\"]?(\w+)['\"]?", custom_query, re.IGNORECASE)
            if col_match:
                col_input = col_match.group(1)
                col = find_column(col_input, df_processed.columns)
                if col and pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[f'{col}_sqrt'] = np.sqrt(df_processed[col].clip(lower=0))
                    st.success(f"Applied square root transform to column '{col}'. New column: '{col}_sqrt'")
                elif col:
                    st.error(f"Column '{col}' not numeric for square root transform.")
                else:
                    st.error(f"Column '{col_input}' not found. Available columns: {', '.join(df_processed.columns)}")
        
        
        # Normalize column
        elif "normalize column" in query_lower:
            col_match = re.search(r"(?:column|col)\s+['\"]?(\w+)['\"]?", custom_query, re.IGNORECASE)
            if col_match:
                col_input = col_match.group(1)
                col = find_column(col_input, df_processed.columns)
                if col and pd.api.types.is_numeric_dtype(df_processed[col]):
                    col_min = df_processed[col].min()
                    col_max = df_processed[col].max()
                    if col_max != col_min:  # Avoid division by zero
                        df_processed[f'{col}_normalized'] = (df_processed[col] - col_min) / (col_max - col_min)
                        st.success(f"Normalized column '{col}'. New column: '{col}_normalized'")
                    else:
                        st.warning(f"Column '{col}' has constant values, cannot normalize.")
                elif col:
                    st.error(f"Column '{col}' not numeric for normalization.")
                else:
                    st.error(f"Column '{col_input}' not found. Available columns: {', '.join(df_processed.columns)}")
        
        # If no patterns matched
        else:
            st.warning(f"Custom preprocessing command not recognized: '{custom_query}'. Supported commands: log transform, drop column, rename column, bin column, square root transform.")
            
    except Exception as e:
        st.error(f"Error in custom preprocessing: {e}")
    
    return df_processed

# === Preprocessing Suggestion Tool ===
def PreprocessingSuggestionTool(df: pd.DataFrame) -> Dict[str, str]:
    """Suggest preprocessing techniques based on dataset characteristics. 
       NOTE: This is mostly superseded by CombinedAnalysisAgent for initial load.
       Retained for targeted calls if necessary, or if CombinedAnalysisAgent fails.
    """
    # This function might be simplified or removed if CombinedAnalysisAgent is robust.
    # For now, let's assume it could be a fallback or used independently.
    dataset_hash_query = f"preprocessing_suggestions_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, dict): return cached_result

    missing = df.isnull().sum()
    total_rows = len(df)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    suggestions = {}
    if missing.sum() > 0:
        missing_pct = missing / total_rows * 100
        for col, pct in missing_pct.items():
            if pct > 0 and col in num_cols:
                suggestions[f"impute_{col}"] = f"Impute missing values in '{col}' with {'mean' if pct < 10 else 'median'} (missing: {pct:.1f}%)."
            elif pct > 0 and col in cat_cols:
                suggestions[f"impute_{col}"] = f"Impute missing values in '{col}' with most frequent value (missing: {pct:.1f}%)."
    
    if len(cat_cols) > 0:
        suggestions["encode_categorical"] = f"Encode {len(cat_cols)} categorical columns ({', '.join(cat_cols)}) for analysis."
    
    if len(num_cols) > 0 and df[num_cols].std().max() > 10: # Heuristic for scaling
        suggestions["scale_features"] = "Scale numerical features to normalize large value ranges."
    
    newline = '\n' # For f-string compatibility
    suggestions_str = "".join([f'- {desc}{newline}' for desc in suggestions.values()])
    prompt = (
        f"Dataset: {total_rows} rows, {len(df.columns)} columns{newline}"
        f"Columns: {', '.join(df.columns)}{newline}"
        f"Data types: {df.dtypes.to_dict()}{newline}"
        f"Missing values: {missing.to_dict()}{newline}"
        f"Existing Suggestions (based on heuristics):{newline}{suggestions_str}"
        f"Based on the above, provide a brief overall 'explanation' text (2-3 sentences) for why these general types of preprocessing steps are recommended for this dataset.{newline}"
        f"Return ONLY the explanation string, no other text, no JSON."
    )
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Provide concise explanations for preprocessing steps."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature
            max_tokens=200 # Reduced max_tokens for suggestions
        )
        explanation_text = response.choices[0].message.content.strip()
        suggestions["explanation"] = explanation_text
        cache_result(dataset_hash_query, prompt, suggestions)
        return suggestions
    except Exception as e:
        st.error(f"Error in PreprocessingSuggestionTool (LLM part): {e}")
        suggestions["explanation"] = "Could not generate LLM explanation due to an error."
        # Cache anyway with the error in explanation
        cache_result(dataset_hash_query, prompt, suggestions)
        return suggestions

# === Visualization Suggestion Tool ===
def VisualizationSuggestionTool(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Suggest visualizations based on dataset characteristics.
       NOTE: This is mostly superseded by CombinedAnalysisAgent for initial load.
       Retained for targeted calls if necessary, or if CombinedAnalysisAgent fails.
    """
    dataset_hash_query = f"visualization_suggestions_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, list): return cached_result

    suggestions = [] # Start with an empty list for LLM to populate primarily
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns

    # Minimal hardcoded suggestions as fallback or seed if LLM fails badly
    if len(cat_cols) > 0:
        # Use Plotly for interactive bar charts
        suggestions.append({
            "query": f"Show bar chart of counts for {cat_cols[0]} using Plotly",
            "desc": f"Interactive bar chart of value counts for categorical column '{cat_cols[0]}'."
        })
    if len(num_cols) > 0:
        # Use Seaborn for distribution analysis
        suggestions.append({
            "query": f"Show histogram of {num_cols[0]} using Seaborn",
            "desc": f"Statistical histogram of numerical column '{num_cols[0]}' to show distribution."
        })
    if len(num_cols) > 1:
        # Use Seaborn for correlation heatmap
        suggestions.append({
            "query": "Show correlation heatmap using Seaborn",
            "desc": "Correlation matrix heatmap for numerical columns."
        })
    
    prompt = f"""
    Dataset: {len(df)} rows, {len(df.columns)} columns
    Columns: {', '.join(df.columns)}
    Data types: {df.dtypes.to_dict()}
    Based on the dataset characteristics, suggest 3-4 diverse and relevant visualizations.
    Consider different visualization libraries and their strengths:
    - Plotly: For interactive charts (bar, line, pie, scatter), large datasets, user exploration
    - Seaborn: For statistical plots (heatmaps, boxplots, violin plots, kde plots, correlation matrices)
    - Matplotlib: For simple static plots when interactivity is not needed
    
    Format as a list of JSON objects. Each object must have a "query" field (a natural language query for a visualization, e.g., "Show bar chart of counts for columnName using Plotly") and a "desc" field (a human-readable description, e.g., "Interactive bar chart of value counts for categorical column 'columnName'.").
    Include the preferred library in the query when relevant (e.g., "using Plotly", "using Seaborn", "using Matplotlib").
    
    Prioritize:
    - Plotly for interactive charts and large datasets
    - Seaborn for statistical analysis (correlation, distribution, categorical comparisons)
    - Matplotlib for basic static visualizations
    
    Return ONLY the list of JSON objects, as a valid JSON array string. No other text before or after.
    Example: [ {{"query": "Show histogram of age using Seaborn", "desc": "Statistical histogram of age distribution"}}, {{"query": "Show interactive scatter plot of height vs weight using Plotly", "desc": "Interactive scatter plot showing relationship between height and weight"}} ]
    """
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Suggest concise, relevant visualizations as a JSON list with appropriate library selections."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature
            max_tokens=512 # Max tokens for suggestions
        )
        content = response.choices[0].message.content
        llm_suggestions = json.loads(content) # Expecting a list directly if model/API supports it well
        if isinstance(llm_suggestions, list):
            # Further validation for individual items can be added here
            cache_result(dataset_hash_query, prompt, llm_suggestions)
            return llm_suggestions[:4] # Limit to 4 suggestions
        else:
            st.warning("LLM did not return a list for visualization suggestions. Using fallback.")
            cache_result(dataset_hash_query, prompt, suggestions[:4]) # Cache fallback
            return suggestions[:4] # Fallback to hardcoded or minimal

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from LLM for visualization suggestions: {e}. Response: {content[:200]}")
        cache_result(dataset_hash_query, prompt, suggestions[:4]) # Cache fallback
        return suggestions[:4]
    except Exception as e:
        st.error(f"Error in VisualizationSuggestionTool (LLM part): {e}")
        cache_result(dataset_hash_query, prompt, suggestions[:4]) # Cache fallback
        return suggestions[:4]

# === Model Recommendation Tool ===
def ModelRecommendationTool(df: pd.DataFrame) -> str:
    """Recommend ML models for the dataset.
       NOTE: This is mostly superseded by CombinedAnalysisAgent for initial load.
       Retained for targeted calls if necessary, or if CombinedAnalysisAgent fails.
    """
    dataset_hash_query = f"model_recommendations_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, str): return cached_result

    num_rows, num_cols = df.shape
    target_col = None
    # Simple heuristic for inferring a potential target column
    for col in df.columns:
        if df[col].nunique() < num_rows * 0.1 and df[col].nunique() > 1: # Avoid constant columns
            target_col = col
            break
    
    task = "classification" if target_col and df[target_col].dtype == 'object' else "regression"
    if not target_col: # If no clear target, suggest clustering
        task = "clustering"

    size = "small" if num_rows < 1000 else "large"
    
    # Basic recommendations based on heuristics
    recommendations_heuristic = []
    if task == "classification":
        recommendations_heuristic.append(("Logistic Regression", "Suitable for categorical outcomes."))
        recommendations_heuristic.append(("Random Forest Classifier", "Handles complex feature interactions."))
        if size == "large":
            recommendations_heuristic.append(("XGBoost Classifier", "High performance for large datasets."))
    elif task == "regression":
        recommendations_heuristic.append(("Linear Regression", "Simple for continuous outcomes."))
        recommendations_heuristic.append(("Random Forest Regressor", "Captures non-linear trends."))
        if size == "large":
            recommendations_heuristic.append(("Gradient Boosting Regressor", "Effective for complex patterns."))
    elif task == "time-series forecasting": # This case was added in the previous step
        recommendations_heuristic.append(("ARIMA / SARIMA", "Classical models for time series data with auto-correlation and seasonality."))
        recommendations_heuristic.append(("Prophet", "Facebook's model, good for time series with strong seasonality, holidays, and missing data."))
        if size == "large":
            recommendations_heuristic.append(("LSTM / GRU (Deep Learning)", "Can capture complex temporal dependencies in large datasets, requires more data and tuning."))
    else: # Clustering (default or if no clear target)
        recommendations_heuristic.append(("K-Means Clustering", "Partitions data into K distinct clusters based on feature similarity."))
        recommendations_heuristic.append(("DBSCAN", "Density-based clustering, good for discovering clusters of varying shapes and handling noise."))
        if size == "large": 
            recommendations_heuristic.append(("Agglomerative Hierarchical Clustering", "Creates a hierarchy of clusters, can be visualized as a dendrogram (can be slow for very large datasets)."))
    
    # TODO: For anomaly detection tasks (if inferred or requested), suggest models like Isolation Forest, One-Class SVM, Autoencoders.

    newline = '\n' # For f-string compatibility
    prompt = f"""
    Dataset: {num_rows} rows, {num_cols} columns.
    Inferred Task: {task} (Target column: {target_col if target_col and task not in ['clustering', 'time-series forecasting'] else 'None directly applicable or task does not require a single target'}).
    Heuristic Model Suggestions:
    {"".join([f'- {model}: {reason}{newline}' for model, reason in recommendations_heuristic])}
    Based on the dataset characteristics and inferred task, provide a brief (2-4 sentences) explanation and recommendation for suitable Machine Learning models.
    Focus on why these types of models are appropriate. You can refine or confirm the heuristic suggestions.
    If task is 'time-series forecasting', mention considerations like data stationarity, seasonality, and potential for exogenous variables if applicable from column names.
    If task is 'clustering', briefly mention the importance of choosing the number of clusters (for K-Means) or parameters like epsilon (for DBSCAN).
    If task is 'classification' or 'regression', briefly mention feature importance or model interpretability if relevant.
    Return ONLY the textual explanation. No JSON, no list, just the paragraph.
    """
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Provide concise model recommendations and explanations."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature for more deterministic code
            max_tokens=256 # Max tokens for this specific output
        )
        recommendation_text = response.choices[0].message.content.strip()
        cache_result(dataset_hash_query, prompt, recommendation_text)
        return recommendation_text
    except Exception as e:
        st.error(f"Error in ModelRecommendationTool (LLM part): {e}")
        # Fallback to a simple heuristic string if LLM fails
        fallback_text = f"Based on the task ({task}), consider models like {', '.join([r[0] for r in recommendations_heuristic])}. LLM explanation failed."
        cache_result(dataset_hash_query, prompt, fallback_text)
        return fallback_text

# === QueryUnderstandingTool ===
def QueryUnderstandingTool(query: str) -> Tuple[str, bool]:
    """Classify query as preprocessing, visualization, or analytics. Returns (intent, is_plotly)."""
    # No separate caching here as it's called frequently and should be very fast.
    
    # First check for explicit library requests
    query_lower = query.lower()
    if "using matplotlib" in query_lower:
        return ("visualization", False)
    if "using seaborn" in query_lower:
        return ("visualization", False)
    if "using plotly" in query_lower:
        return ("visualization", True)
    
    # Check for custom preprocessing prefix first
    if query_lower.startswith("custom preprocess:"):
        return ("custom_preprocessing", False)

    # Always treat correlation matrix/heatmap as visualization
    corr_viz_patterns = [
        "correlation matrix", "correlation heatmap", "corr heatmap", "corr matrix"
    ]
    if any(pattern in query_lower for pattern in corr_viz_patterns):
        return ("visualization", False)  # Seaborn is better for correlation heatmaps
    
    analytical_patterns = [
        "show missing", "missing value", "missing data", "check missing",
        "duplicates", "duplicate", "is there any", "how many",
        "summary", "describe", "info", "statistics", "stats",
        "count", "unique", "nunique", "shape", "size",
        "correlation", "corr", "distribution"
    ]
    
    # Check for visualization patterns
    viz_patterns = [
        "plot", "chart", "graph", "visualize", "histogram", "scatter",
        "bar chart", "pie chart", "line chart", "heatmap", "boxplot", "violin plot"
    ]
    
    # Check for preprocessing patterns (only if not analytical)
    preprocessing_patterns = [
        "impute", "encode", "scale", "normalize", "preprocess",
        "fill missing", "handle missing", "transform", "convert"
    ]
    
    # Prioritize analytical queries
    if any(pattern in query_lower for pattern in analytical_patterns):
        intent = "analytics"
        is_plotly = False
    elif any(pattern in query_lower for pattern in viz_patterns):
        intent = "visualization"
        # Determine if Plotly should be used based on query patterns
        plotly_patterns = ["interactive", "3d", "large dataset", "explore"]
        seaborn_patterns = ["heatmap", "boxplot", "violin", "kde", "distribution", "statistical"]
        
        if any(pattern in query_lower for pattern in plotly_patterns):
            is_plotly = True
        elif any(pattern in query_lower for pattern in seaborn_patterns):
            is_plotly = False  # Use Seaborn for statistical plots
        else:
            # Default decision: use Plotly for bar, scatter, line charts for interactivity
            interactive_chart_patterns = ["bar chart", "scatter", "line chart", "pie chart"]
            is_plotly = any(pattern in query_lower for pattern in interactive_chart_patterns)
    elif any(pattern in query_lower for pattern in preprocessing_patterns):
        intent = "preprocessing"
        is_plotly = False
    else:
        # Use LLM as fallback for unclear cases
        messages = [
            {"role": "system", "content": "Classify the user query. Respond with one word: 'preprocessing', 'visualization', 'plotly', or 'analytics'. 'plotly' is for interactive visualizations or when user requests Plotly. If it's a statistical plot (heatmap, boxplot, distribution), use 'visualization'. Detailed thinking off."},
            {"role": "user", "content": query}
        ]
        
        try:
            response = client.chat.completions.create(
                model=QUERY_CLASSIFICATION_MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=10
            )
            
            intent = response.choices[0].message.content.strip().lower()
            valid_intents = ["preprocessing", "visualization", "plotly", "analytics"]
            if intent not in valid_intents:
                intent = "analytics"  # Default to analytics for unclear queries
            
            is_plotly = intent == "plotly"
            if is_plotly:
                intent = "visualization"
        except Exception as e:
            st.error(f"Error in QueryUnderstandingTool: {e}. Defaulting intent.")
            intent = "analytics"  # Default to analytics on error
            is_plotly = False

    return (intent, is_plotly)

# === Code Generation Tools ===
def PlotCodeGeneratorTool(cols: List[str], query: str, data_size: int = 0) -> str:
    """Generate a prompt for optimized pandas+matplotlib/seaborn code with intelligent library selection."""
    
    # Intelligent library selection based on query patterns
    query_lower = query.lower()
    
    # Detect if Seaborn should be used for statistical plots
    seaborn_patterns = [
        "heatmap", "correlation", "boxplot", "violin", "kde", "distribution",
        "statistical", "seaborn", "box plot", "violin plot"
    ]
    use_seaborn = any(pattern in query_lower for pattern in seaborn_patterns)
    
    # Determine optimization strategy based on data size
    performance_opts = ""
    if data_size > 1000:
        performance_opts = """
    PERFORMANCE OPTIMIZATIONS for large dataset:
    - Use smaller figure sizes: figsize=(8, 6) instead of larger
    - Reduce marker sizes in scatter plots: s=1 for >5000 points
    - Add transparency: alpha=0.6 for overlapping points
    - Use plt.tight_layout() for better memory usage
    - For Seaborn plots, consider using sample data if >10000 points
        """
    
    library_instruction = ""
    if use_seaborn:
        library_instruction = """
    LIBRARY SELECTION: Use Seaborn (sns) for this statistical visualization.
    - Import seaborn as sns (already available)
    - Use appropriate Seaborn functions: sns.heatmap(), sns.boxplot(), sns.violinplot(), sns.histplot(), etc.
    - Set matplotlib figure: fig, ax = plt.subplots(figsize=(10, 8))
    - Pass ax=ax to Seaborn functions for consistency
    - Use plt.tight_layout() before returning the figure
        """
    else:
        library_instruction = """
    LIBRARY SELECTION: Use Matplotlib for this visualization.
    - Use matplotlib.pyplot (as plt) for plotting
    - Create figure: fig, ax = plt.subplots(figsize=(8, 6))
    - Use standard matplotlib functions: ax.plot(), ax.scatter(), ax.bar(), etc.
        """
    
    example_code = ""
    if use_seaborn:
        example_code = """
    Example for Seaborn statistical plot:
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Data processing...
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Statistical Visualization')
    plt.tight_layout()
    result = fig
    ```"""
    else:
        example_code = """
    Example for Matplotlib plot:
    ```python
    import matplotlib.pyplot as plt
    # Data processing...
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(df) > 5000:
        ax.scatter(df['x'], df['y'], s=1, alpha=0.6)  # Optimized for large data
    else:
        ax.scatter(df['x'], df['y'], s=20, alpha=0.8)
    ax.set_title('Data Visualization')
    plt.tight_layout()
    result = fig
    ```"""
    
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas and the appropriate visualization library to answer:
    "{query}"
    Dataset size: {data_size} rows
    
    {library_instruction}
    
    {performance_opts}
    
    Rules:
    1. Use pandas for data manipulation and the selected library for plotting.
    2. For large datasets (>1000 rows), add performance optimizations.
    3. Assign the final result (matplotlib Figure) to `result`.
    4. Create ONE plot with appropriate size, add title/labels.
    5. Return inside a single ```python fence.
    
    {example_code}
    """

def PlotlyCodeGeneratorTool(cols: List[str], query: str, data_size: int = 0) -> str:
    """Generate a prompt for optimized Plotly code for interactive visualizations."""
    
    # Determine optimization strategy based on data size
    performance_opts = ""
    if data_size > 1000:
        performance_opts = """
    PERFORMANCE OPTIMIZATIONS for large dataset:
    - Use plotly.graph_objects.Scattergl for scatter plots with >1000 points (WebGL rendering)
    - Disable hover information for datasets >5000 points: hoverinfo='skip'
    - Reduce marker size for large datasets: marker=dict(size=3)
    - For line plots, use 'lines' mode without markers for >1000 points
    - Consider data sampling or aggregation for extremely large datasets (>50000 points)
        """
    else:
        performance_opts = """
    PERFORMANCE OPTIMIZATIONS:
    - Standard Plotly rendering is sufficient for datasets under 1000 rows
    - Enable full hover information and interactivity
    - Use normal marker sizes for better visibility
        """
    
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas and Plotly to create an interactive visualization for:
    "{query}"
    Dataset size: {data_size} rows
    
    {performance_opts}
    
    Rules:
    1. Use plotly.express (px) for simple plots or plotly.graph_objects (go) for advanced customization
    2. For large datasets (>1000 rows), use performance optimizations
    3. Assign the final result (Plotly Figure) to `result`
    4. Create an interactive plot with appropriate titles, labels, and hover information
    5. Return inside a single ```python fence
    
    Example for performance-optimized Plotly:
    ```python
    import plotly.express as px
    import plotly.graph_objects as go
    # Data processing...
    if len(df) > 1000:
        # Use WebGL for better performance
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=df['x'], y=df['y'], 
            mode='markers',
            marker=dict(size=3),
            hoverinfo='skip' if len(df) > 5000 else 'x+y'
        ))
    else:
        # Standard plotly express for smaller datasets
        fig = px.scatter(df, x='x', y='y', title='Interactive Visualization')
    
    fig.update_layout(title='Interactive Data Visualization')
    result = fig
    ```
    """

def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for pandas-only code."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas to answer: "{query}"
    
    Rules:
    1. Use pandas operations on `df` only (no plotting libraries).
    2. For missing values queries: use df.isnull().sum(), df.info(), or create summary DataFrames.
    3. For duplicates queries: use df.duplicated().sum(), df[df.duplicated()], or df.drop_duplicates().
    4. For statistical queries: use df.describe(), df.nunique(), df.value_counts(), etc.
    5. For data info queries: use df.shape, df.dtypes, df.columns, etc.
    6. Assign the final result to `result` variable.
    7. If creating a summary, make it a DataFrame or Series for better display.
    8. Return inside a single ```python fence.
    
    Examples:
    - For "show missing values": result = df.isnull().sum().to_frame(name='Missing_Count')
    - For "check duplicates": result = f"Total duplicates: {{df.duplicated().sum()}}"
    - For "data summary": result = df.describe()
    """

def PreprocessingCodeGeneratorTool(cols: List[str], query: str) -> Tuple[str, Dict]:
    """Generate preprocessing parameters and code from query."""
    params = {
        "missing_strategy": "None", 
        "encode_categorical": False, 
        "scale_features": False, 
        "target_columns": None, # Initialize as None
        "scaling_strategy": "standard", 
        "constant_value_impute": None,
        "one_hot_encode_columns": None, # Initialize as None
        "outlier_strategy": None,
        "outlier_columns": None,
        "feature_engineering": None
    }
    query_lower = query.lower()

    # If the query is about correlation matrix/heatmap, do not generate preprocessing code
    corr_viz_patterns = [
        "correlation matrix", "correlation heatmap", "corr heatmap", "corr matrix"
    ]
    if any(pattern in query_lower for pattern in corr_viz_patterns):
        return "", params

    impute_match = re.match(r"impute column '([^']+)' with (mean|median|mode|constant|forward_fill|backward_fill)(?: \(value: (.+)\))?", query_lower)
    encode_match = re.match(r"(label_encoding|one_hot_encoding) for column '([^']+)'", query_lower)
    scale_match = re.match(r"(standard_scaling|min_max_scaling|robust_scaling) for columns: (.+)", query_lower)
    outlier_match = re.match(r"apply (remove|cap) outlier handling to columns: (.+)", query_lower)
    poly_match = re.match(r"add polynomial features \(degree (\d+)\) for columns: (.+)", query_lower)
    date_match = re.match(r"extract date components \(year, month, day\) from columns: (.+)", query_lower)

    action_taken = False
    if impute_match:
        action_taken = True
        params["target_columns"] = [impute_match.group(1)]
        strategy = impute_match.group(2)
        params["missing_strategy"] = strategy # PreprocessingTool will map ffill/bfill if needed
        if strategy == "constant" and impute_match.group(3):
            params["constant_value_impute"] = impute_match.group(3)
        # For mean, median, mode, PreprocessingTool will handle based on missing_strategy and target_columns
    elif encode_match:
        action_taken = True
        strategy = encode_match.group(1)
        column = encode_match.group(2)
        params["target_columns"] = [column]
        if strategy == "label_encoding":
            params["encode_categorical"] = True # PreprocessingTool will apply LE to target_columns if cat
        elif strategy == "one_hot_encoding":
            params["one_hot_encode_columns"] = [column]
            params["encode_categorical"] = True # Also set this, PreprocessingTool can decide not to LE if OHE is done
    elif scale_match:
        action_taken = True
        strategy = scale_match.group(1)
        columns_str = scale_match.group(2)
        params["target_columns"] = [col.strip() for col in columns_str.split(',')]
        params["scale_features"] = True
        params["scaling_strategy"] = strategy.replace("_scaling", "") # e.g. "standard_scaling" -> "standard"
    elif outlier_match:
        action_taken = True
        strategy = outlier_match.group(1)
        columns_str = outlier_match.group(2)
        params["outlier_strategy"] = strategy
        params["outlier_columns"] = [col.strip() for col in columns_str.split(',')]
    elif poly_match:
        action_taken = True
        degree = int(poly_match.group(1))
        columns_str = poly_match.group(2)
        params["feature_engineering"] = {
            "polynomial_cols": [col.strip() for col in columns_str.split(',')],
            "polynomial_degree": degree
        }
    elif date_match:
        action_taken = True
        columns_str = date_match.group(1)
        params["feature_engineering"] = {
            "date_cols": [col.strip() for col in columns_str.split(',')]
        }
    else:
        # Fallback to original NLP-based parameter detection
        if "impute" in query_lower:
            action_taken = True
            if "mean" in query_lower: params["missing_strategy"] = "mean"
            elif "median" in query_lower: params["missing_strategy"] = "median"
            elif "most frequent" in query_lower or "mode" in query_lower: params["missing_strategy"] = "most_frequent"
            elif "forward fill" in query_lower or "ffill" in query_lower: params["missing_strategy"] = "forward_fill"
            elif "backward fill" in query_lower or "bfill" in query_lower: params["missing_strategy"] = "backward_fill"
        if "encode" in query_lower or "categorical" in query_lower:
            action_taken = True
            params["encode_categorical"] = True
            # Could try to extract target columns from NLP here if desired
        if "scale" in query_lower or "normalize" in query_lower:
            action_taken = True
            params["scale_features"] = True
            # Could try to extract target columns/scaling strategy from NLP here

    # If no specific preprocessing action identified by query, it might be an analytical query misclassified
    # or a general request. For safety, if no flags are set, don't generate PreprocessingTool call.
    if not action_taken and not (params["missing_strategy"] != "None" or params["encode_categorical"] or params["scale_features"]):
        # This query is likely not a preprocessing task for this tool.
        # Return empty code and default params, CodeGenerationAgent will decide what to do.
        return "", params 

    # Construct the call to PreprocessingTool
    # df_processed = PreprocessingTool(df, missing_strategy='...', ...)
    # result = df_processed
    code_lines = [
        "df_processed = PreprocessingTool(",
        "    df=df,",
        f"    missing_strategy='{params['missing_strategy']}',",
        f"    encode_categorical={params['encode_categorical']},",
        f"    scale_features={params['scale_features']},",
        f"    target_columns={params['target_columns']},",
        f"    scaling_strategy='{params['scaling_strategy']}',",
        f"    constant_value_impute={repr(params['constant_value_impute'])},",
        f"    one_hot_encode_columns={params['one_hot_encode_columns']},",
        f"    outlier_strategy={repr(params['outlier_strategy'])},",
        f"    outlier_columns={params['outlier_columns']},",
        f"    feature_engineering={params['feature_engineering']}",
        ")",
        "result = df_processed"
    ]
    final_code = "\n".join(code_lines)
    
    return final_code, params

# === CodeGenerationAgent ===
def CodeGenerationAgent(query: str, df: pd.DataFrame):
    """Generate code using LLM, with caching and visualization optimization."""
    # Cache key combines query and DataFrame hash for context-specific code
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cached_code, cached_result_obj = get_cached_result(query_hash)
    
    if cached_code: # If we have cached code for this query
        intent, is_plotly = QueryUnderstandingTool(query)
        return cached_code, intent, is_plotly, cached_result_obj

    intent, is_plotly = QueryUnderstandingTool(query)
    code = ""
    
    if intent == "preprocessing":
        # PreprocessingCodeGeneratorTool generates code directly, not an LLM prompt
        code, _ = PreprocessingCodeGeneratorTool(df.columns.tolist(), query)
    elif intent == "custom_preprocessing":
        # Handle custom preprocessing queries using CustomPreprocessingTool
        custom_query = query.replace("Custom preprocess:", "").strip()
        code = f"""result = CustomPreprocessingTool(df, "{custom_query}")"""
    elif intent in ["visualization"]:
        # === VISUALIZATION OPTIMIZATION FOR LARGE DATASETS ===
        df_viz = df  # Default to original dataframe
        viz_metadata = None
        
        # Check if dataset is large and needs optimization
        if len(df) > 10000:
            # Get visualization parameters from session state or use defaults
            viz_params = {
                "query": query,
                "sample_rate": getattr(st.session_state, 'viz_sample_rate', 0.1),
                "max_rows": 10000,
                "agg_type": getattr(st.session_state, 'viz_agg_type', 'mean'),
                "optimization_enabled": getattr(st.session_state, 'viz_optimization_enabled', True)
            }
            
            if viz_params["optimization_enabled"]:
                # Check cache for optimized visualization data
                df_hash = get_df_hash(df)
                cached_viz_df, cached_metadata = get_cached_visualization_data(df_hash, viz_params)
                
                if cached_viz_df is not None:
                    df_viz = cached_viz_df
                    viz_metadata = cached_metadata
                    st.info(f"🚀 Using cached optimized visualization data ({cached_metadata['final_rows']} rows, {cached_metadata['data_reduction']:.1%} reduction)")
                else:
                    # Apply sampling or aggregation
                    df_viz, viz_metadata = sample_or_aggregate_df(
                        df, 
                        max_rows=viz_params["max_rows"],
                        sample_rate=viz_params["sample_rate"],
                        agg_type=viz_params["agg_type"]
                    )
                    
                    # Cache the optimized data
                    cache_visualization_data(df_hash, viz_params, df_viz, viz_metadata)
                    
                    # Show user feedback
                    if viz_metadata["operation"] != "none":
                        st.warning(f"⚡ Large dataset optimization: {viz_metadata['operation']} applied. "
                                 f"Reduced from {viz_metadata['original_rows']:,} to {viz_metadata['final_rows']:,} rows "
                                 f"({viz_metadata['data_reduction']:.1%} reduction)")
        
        # === INTELLIGENT LIBRARY SELECTION ===
        query_lower = query.lower()
        
        # Determine which library to use based on query patterns and dataset characteristics
        use_plotly = False
        library_reason = ""
        
        # Check for explicit library requests first
        if "using plotly" in query_lower:
            use_plotly = True
            library_reason = "Explicitly requested Plotly"
        elif "using seaborn" in query_lower or "using matplotlib" in query_lower:
            use_plotly = False
            library_reason = f"Explicitly requested {'Seaborn' if 'seaborn' in query_lower else 'Matplotlib'}"
        else:
            # Automatic library selection based on criteria
            plotly_triggers = [
                len(df_viz) > 1000,  # Large datasets benefit from Plotly's performance
                "interactive" in query_lower,
                "3d" in query_lower,
                "explore" in query_lower,
                any(word in query_lower for word in ["bar chart", "scatter", "line chart", "pie chart"]) and "heatmap" not in query_lower
            ]
            
            seaborn_triggers = [
                "heatmap" in query_lower,
                "correlation" in query_lower,
                "boxplot" in query_lower or "box plot" in query_lower,
                "violin" in query_lower,
                "kde" in query_lower,
                "distribution" in query_lower,
                "statistical" in query_lower
            ]
            
            if any(seaborn_triggers):
                use_plotly = False
                library_reason = "Statistical plot detected - using Seaborn"
            elif any(plotly_triggers):
                use_plotly = True
                library_reason = f"Interactive/large dataset visualization - using Plotly (dataset: {len(df_viz)} rows)"
            else:
                # Default to Matplotlib for simple static plots
                use_plotly = False
                library_reason = "Simple static plot - using Matplotlib"
        
        # Override the is_plotly from QueryUnderstandingTool with our intelligent selection
        is_plotly = use_plotly
        
        # Generate optimized code with dataset size information
        prompt_template_func = PlotlyCodeGeneratorTool if is_plotly else PlotCodeGeneratorTool
        prompt = prompt_template_func(df_viz.columns.tolist(), query, len(df_viz))
        
        system_message_content = "You are an expert Python programmer. Write clean, efficient code. "
        if is_plotly:
            system_message_content += f"Generate Plotly code for interactive visualization. {library_reason}. Return ONLY a single Python code block (```python ... ```)."
        else:
            system_message_content += f"Generate Matplotlib/Seaborn code for visualization. {library_reason}. Return ONLY a single Python code block (```python ... ```)."

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=POWERFUL_MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=1024
        )
        code = extract_first_code_block(response.choices[0].message.content)
        
        # Modify the code to use the optimized dataframe
        if viz_metadata and viz_metadata["operation"] != "none":
            # Replace 'df' with the optimized dataframe in the generated code
            # This is a simple approach - in practice, you might want more sophisticated code modification
            code = f"# Using optimized dataset ({viz_metadata['final_rows']} rows)\ndf_viz = df  # This will be replaced with optimized data\n" + code.replace("df", "df_viz")
            
        # Store visualization metadata and dataframe for ExecutionAgent
        if hasattr(st, 'session_state'):
            st.session_state._last_viz_metadata = viz_metadata
            st.session_state._last_df_viz = df_viz if viz_metadata and viz_metadata["operation"] != "none" else None
            st.session_state._last_library_reason = library_reason
    else: # intent == "analytics"
        # Check for common analytical queries and provide hardcoded solutions
        query_lower = query.lower()
        if "missing" in query_lower and ("show" in query_lower or "check" in query_lower):
            # Hardcoded solution for missing values
            code = """missing_counts = df.isnull().sum()
missing_df = pd.DataFrame({
    'Column': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Missing_Percentage': (missing_counts.values / len(df)) * 100
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
result = missing_df if not missing_df.empty else "No missing values found in the dataset." """
        elif "duplicate" in query_lower:
            # Hardcoded solution for duplicates
            code = """duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    duplicate_rows = df[df.duplicated(keep=False)]
    result = f"Found {duplicate_count} duplicate rows. First few duplicates:\\n{duplicate_rows.head().to_string()}"
else:
    result = "No duplicate rows found in the dataset." """
        elif "shape" in query_lower or "size" in query_lower:
            code = """result = f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns" """
        elif "info" in query_lower or "summary" in query_lower:
            code = """result = df.describe()"""
        elif ("correlation" in query_lower and "heatmap" in query_lower) or \
             ("correlation heatmap" in query_lower) or \
             ("correlation matrix heatmap" in query_lower) or \
             ("corr heatmap" in query_lower):
            code = '''
import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = df.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
           square=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=ax)
ax.set_title('Correlation Matrix Heatmap')
plt.tight_layout()
result = fig
'''
        elif "filter rows where" in query_lower:
            # Extract filter condition from query
            filter_pattern = r"filter rows where (\w+) ([><=!]+) (.+)"
            filter_match = re.search(filter_pattern, query_lower)
            if filter_match:
                col = filter_match.group(1)
                op = filter_match.group(2)
                val = filter_match.group(3)
                # Try to convert value to appropriate type
                try:
                    if val.replace('.', '', 1).isdigit():
                        val = float(val) if '.' in val else int(val)
                    else:
                        val = f"'{val}'"
                except:
                    val = f"'{val}'"
                code = f"""filtered_df = df[df['{col}'] {op} {val}]
result = f"Filtered dataset: {{len(filtered_df)}} rows (from {{len(df)}} original rows)"
if len(filtered_df) > 0:
    result += f"\\n\\nFirst 5 rows of filtered data:\\n{{filtered_df.head().to_string()}}"
else:
    result += "\\nNo rows match the filter criteria." """
            else:
                code = """result = "Could not parse filter condition. Use format: 'column operator value'" """
        else:
            # Use LLM for other analytical queries
            prompt = CodeWritingTool(df.columns.tolist(), query)
            messages = [
                {"role": "system", "content": "You are an expert Python programmer. Write clean, efficient pandas code. Return ONLY a single Python code block (```python ... ```). No explanations before or after the code block."},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=POWERFUL_MODEL_NAME, # Use powerful model
                messages=messages,
                temperature=0.1, # Low temperature
                max_tokens=1024
            )
            code = extract_first_code_block(response.choices[0].message.content)
    
    # Cache the generated code. The execution result will be cached by ExecutionAgent later if successful.
    # Store an empty string or a placeholder indicating "not executed yet" for the result part of this cache entry.
    cache_result(query_hash, code, "__CODE_GENERATED_NOT_EXECUTED__") 
    return code, intent, is_plotly, "__CODE_GENERATED_NOT_EXECUTED__" # Return a placeholder for result

# === ExecutionAgent ===
def ExecutionAgent(code: str, df: pd.DataFrame, intent: str, is_plotly: bool, query_for_cache: str, df_viz: pd.DataFrame = None):
    """Execute code safely and return result. Updates cache with execution result. Handles optimized visualization data."""
    env = {"pd": pd, "df": df, "PreprocessingTool": PreprocessingTool, "CustomPreprocessingTool": CustomPreprocessingTool, "np": np}
    
    # Add optimized visualization dataframe if provided
    if df_viz is not None:
        env["df_viz"] = df_viz
        # If the code uses df_viz, ensure it's available
        if "df_viz" in code:
            env["df"] = df_viz  # Replace df with optimized version for visualization
    
    if intent in ["visualization"]:
        plt.rcParams["figure.dpi"] = 100
        env["plt"] = plt
        env["sns"] = sns  # Add Seaborn support
        env["px"] = px    # Add Plotly Express
        env["go"] = go    # Add Plotly Graph Objects
        env["io"] = io
    
    query_hash = hashlib.md5(query_for_cache.encode()).hexdigest()

    try:
        # Debug: Print the code being executed if there might be issues
        if "df_processed" in code and "PreprocessingTool" in code:
            print(f"DEBUG: Executing preprocessing code:\n{code}")
        elif "df_viz" in code:
            print(f"DEBUG: Executing optimized visualization code for {len(df_viz) if df_viz is not None else 'unknown'} rows")
        
        exec(code, {}, env)
        result = env.get("result", None)
        
        # Successfully executed, cache the result along with the code
        cache_result(query_hash, code, result)
        
        # Handle different types of visualization results
        if hasattr(result, '__module__') and result.__module__ and 'plotly' in result.__module__:
            # This is a Plotly figure
            return result
        elif isinstance(result, (plt.Figure, plt.Axes)):
            # This is a Matplotlib/Seaborn figure
            return result
        else:
            # For non-visualization results or other data types
            return result
            
    except Exception as exc:
        import traceback
        error_details = traceback.format_exc()
        
        # Enhanced error message with code context
        error_message = f"Error executing code: {exc}\n\nCode that failed:\n{code}\n\nFull traceback:\n{error_details}"
        
        # Print to console for debugging
        print(f"EXECUTION ERROR for query '{query_for_cache}':")
        print(f"Code:\n{code}")
        print(f"Error: {exc}")
        
        # Cache the error message as the result for this query/code
        cache_result(query_hash, code, error_message)
        return error_message

# === ReasoningCurator ===
def ReasoningCurator(query: str, result: Any) -> str:
    """Build LLM prompt for reasoning about results."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))
    is_chartjs = isinstance(result, dict) and "type" in result and result["type"] in ["bar", "line", "pie"]
    
    if is_error:
        desc = result
    elif is_plot:
        title = result._suptitle.get_text() if isinstance(result, plt.Figure) and result._suptitle else result.get_title() if isinstance(result, plt.Axes) else "Chart"
        desc = f"[Plot Object: {title}]"
    elif is_chartjs:
        title = result.get("options", {}).get("plugins", {}).get("title", {}).get("text", "Chart")
        desc = f"[Chart.js Object: {title}]"
    else:
        desc = str(result)[:300]
    
    prompt = f'''
    The user asked: "{query}".
    Result: {desc}
    Explain in 2–3 concise sentences what this tells about the data (mention charts only if relevant).
    '''
    return prompt

# === ReasoningAgent ===
def ReasoningAgent(query: str, result: Any):
    """Stream LLM reasoning for results."""
    prompt = ReasoningCurator(query, result)
    # This agent is for detailed reasoning, so POWERFUL_MODEL_NAME is appropriate.
    response = client.chat.completions.create(
        model=POWERFUL_MODEL_NAME, 
        messages=[{"role": "system", "content": "detailed thinking on. You are an insightful data analyst."},
                  {"role": "user", "content": prompt}],
        temperature=0.2, # As per original, can be low for factual reasoning
        max_tokens=1024, # As per original
        stream=True
    )
    
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
            if in_think or ("<think>" in full_response and not "</think>" in full_response):
                thinking_content += token
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )
    
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return thinking_content, cleaned

# === DataFrameSummaryTool ===
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate summary prompt for LLM.
       NOTE: This is mostly superseded by CombinedAnalysisAgent for initial load.
    """
    prompt = f"""
    Given a dataset with {len(df)} rows and {len(df.columns)} columns:
    Columns: {', '.join(df.columns)}
    Data types: {df.dtypes.to_dict()}
    Missing values: {df.isnull().sum().to_dict()}
    Provide:
    1. A brief description of what this dataset contains (1-2 sentences).
    2. 3-4 possible data analysis questions that could be asked of this dataset.
    Keep it very concise and focused. Return ONLY the text, no JSON, no markdown.
    """
    return prompt

# === MissingValueSummaryTool ===
def MissingValueSummaryTool(df: pd.DataFrame) -> str:
    """Generate missing value summary prompt for LLM."""
    missing = df.isnull().sum().to_dict()
    total_missing = sum(missing.values())
    prompt = f"""
    Dataset: {len(df)} rows, {len(df.columns)} columns
    Missing values: {missing}
    Total missing: {total_missing}
    Provide a brief summary (2-3 sentences) of missing values and their potential impact on analysis.
    Return ONLY the textual summary. No JSON, no markdown.
    """
    return prompt

# === DataInsightAgent ===
def DataInsightAgent(df: pd.DataFrame) -> str:
    """Generate dataset summary and questions.
       NOTE: This is largely superseded by CombinedAnalysisAgent for initial load.
       Retained for targeted calls or if CombinedAnalysisAgent fails.
    """
    # Use dataset hash for caching key
    dataset_hash_query = f"dataset_insights_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, str): 
        return cached_result
    
    prompt = DataFrameSummaryTool(df)
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Provide brief, focused insights as plain text."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature
            max_tokens=256 # Reduced max_tokens
        )
        
        result = response.choices[0].message.content.strip()
        cache_result(dataset_hash_query, prompt, result) # Cache string result
        return result
    except Exception as e:
        st.error(f"Error in DataInsightAgent: {e}")
        return f"Error generating insights: {e}"

# === MissingValueAgent ===
def MissingValueAgent(df: pd.DataFrame) -> str:
    """Generate missing value summary with LLM insights."""
    dataset_hash_query = f"missing_value_summary_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, str): 
        return cached_result
    
    prompt = MissingValueSummaryTool(df)
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Provide a concise summary of missing values as plain text."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature
            max_tokens=200 # Reduced max_tokens
        )
        
        result = response.choices[0].message.content.strip()
        cache_result(dataset_hash_query, prompt, result)
        return result
    except Exception as e:
        st.error(f"Error in MissingValueAgent: {e}")
        return f"Error generating missing value summary: {e}"

# === Helpers ===
def load_data(uploaded_file) -> pd.DataFrame | None:
    """Loads data from an uploaded file (CSV or Excel) into a pandas DataFrame."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            if hasattr(st, 'error'):  # Only call st.error if Streamlit is available
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        if hasattr(st, 'error'):  # Only call st.error if Streamlit is available
            st.error(f"Error loading file: {e}")
        return None

def extract_first_code_block(text: str) -> str:
    """Extract first Python code block from markdown."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# === Combined Initial Analysis Agent ===
def CombinedAnalysisAgent(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates initial dataset insights, preprocessing suggestions, visualization suggestions, and model recommendations using a single LLM call."""
    
    dataset_hash_query = f"combined_analysis_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, dict):
        # Basic validation if the cached result has expected keys
        if all(k in cached_result for k in ["insights", "preprocessing_suggestions", "visualization_suggestions", "model_recommendations"]):
            return cached_result

    num_rows, num_cols = df.shape
    column_names = ", ".join(df.columns)
    dtypes_dict = df.dtypes.to_dict()
    missing_values_dict = df.isnull().sum().to_dict()

    # Construct a detailed prompt for the LLM
    prompt = f"""
    Analyze the following dataset and provide a comprehensive analysis. The dataset has {num_rows} rows and {num_cols} columns.
    Column Names: {column_names}
    Data Types: {dtypes_dict}
    Missing Values: {missing_values_dict}

    Please provide the output as a single JSON object with the following three top-level keys:
    1.  "insights": A string containing a brief description of what this dataset likely contains and 3-4 possible data analysis questions that could be asked.
    2.  "preprocessing_suggestions": A JSON object. Keys should be descriptive identifiers (e.g., "impute_columnName", "encode_categorical", "scale_features"). Values should be strings explaining the suggestion (e.g., "Impute missing values in 'columnName' with mean (missing: X.X%).", "Encode N categorical columns (col1, col2) for analysis.", "Scale numerical features to normalize large value ranges."). Also include an "explanation" key with a general rationale for the suggested preprocessing steps.
    3.  "visualization_suggestions": A list of JSON objects. Each object should have a "query" field (a natural language query for a visualization, e.g., "Show bar chart of counts for columnName") and a "desc" field (a human-readable description, e.g., "Bar chart of value counts for categorical column 'columnName'."). Suggest 3-4 diverse and relevant visualizations.

    For visualization suggestions, consider different visualization libraries and their strengths:
    - Plotly: For interactive charts (bar, line, pie, scatter), large datasets, user exploration
    - Seaborn: For statistical plots (heatmaps, boxplots, violin plots, kde plots, correlation matrices)
    - Matplotlib: For simple static plots when interactivity is not needed
    
    Include the preferred library in the query when relevant (e.g., "using Plotly", "using Seaborn", "using Matplotlib").
    
    Prioritize:
    - Plotly for interactive charts and large datasets
    - Seaborn for statistical analysis (correlation, distribution, categorical comparisons)
    - Matplotlib for basic static visualizations

    Prioritize concise and actionable information. Ensure the entire output is a single valid JSON object.
    Example for preprocessing_suggestions value:
    {{ "impute_age": "Impute missing values in 'age' with mean (missing: 5.2%).", "encode_gender": "Encode categorical column 'gender'.", "explanation": "Imputation handles missing data, encoding prepares categorical data for modeling."}}
    Example for visualization_suggestions value:
    [ {{"query": "Show histogram of age using Seaborn", "desc": "Statistical histogram of age distribution"}}, {{"query": "Show interactive scatter plot of height vs weight using Plotly", "desc": "Interactive scatter plot showing relationship between height and weight"}} ]
    
    Return ONLY the JSON object. No other text before or after.
    """

    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert data analyst. Provide results as a single, valid JSON object, adhering strictly to the requested structure. No extra text before or after the JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2048 # Increased to accommodate comprehensive JSON output
        )
        
        content = response.choices[0].message.content
        # Attempt to parse the JSON content
        # The LLM should ideally return a string that is parseable into a JSON object.
        # If the API supports a json_object response_format, this will be more reliable.
        parsed_json = json.loads(content)

        # Validate the structure of the parsed_json using Pydantic
        try:
            validated_result = CombinedAnalysisResult(**parsed_json)
            # Convert back to dict for consistency with existing code
            validated_dict = validated_result.model_dump()
            
            # Ensure visualization suggestions are not empty - add fallback if needed
            if not validated_dict.get('visualization_suggestions'):
                # Add basic fallback suggestions based on dataset characteristics
                num_cols = df.select_dtypes(include=['float64', 'int64']).columns
                cat_cols = df.select_dtypes(include=['object']).columns
                
                fallback_suggestions = []
                if len(num_cols) > 0:
                    fallback_suggestions.append({
                        "query": f"Show histogram of {num_cols[0]} using Seaborn",
                        "desc": f"Statistical distribution of {num_cols[0]}"
                    })
                if len(cat_cols) > 0:
                    fallback_suggestions.append({
                        "query": f"Show bar chart of {cat_cols[0]} counts using Plotly",
                        "desc": f"Interactive count distribution of {cat_cols[0]}"
                    })
                if len(num_cols) > 1:
                    fallback_suggestions.append({
                        "query": "Show correlation heatmap using Seaborn",
                        "desc": "Correlation matrix of numerical variables"
                    })
                if len(num_cols) > 0 and len(cat_cols) > 0:
                    fallback_suggestions.append({
                        "query": f"Show boxplot of {num_cols[0]} by {cat_cols[0]} using Seaborn",
                        "desc": f"Distribution of {num_cols[0]} across {cat_cols[0]} categories"
                    })
                
                validated_dict['visualization_suggestions'] = fallback_suggestions[:3]  # Limit to 3
                print("Added fallback visualization suggestions")
            
            # Cache the validated result
            cache_result(dataset_hash_query, prompt, validated_dict)
            return validated_dict
        except Exception as validation_error:
            st.error(f"LLM output validation error for Combined Analysis: {validation_error}")
            # Log the original parsed_json for debugging
            print(f"Original parsed JSON that failed validation: {parsed_json}")
            
            # Basic fallback validation for legacy compatibility
            if not isinstance(parsed_json, dict) or not all(k in parsed_json for k in ["insights", "preprocessing_suggestions", "visualization_suggestions"]):
                # Fallback or error handling if JSON is not as expected
                st.error("LLM returned an unexpected JSON structure for combined analysis. Using fallback structure.")
                return {
                    "insights": f"Error: Could not validate LLM response. Validation error: {validation_error}",
                    "preprocessing_suggestions": {},
                    "visualization_suggestions": [],
                    "model_recommendations": ""
                }
            else:
                # If basic structure is okay but Pydantic validation failed, log and return as-is
                # Ensure model_recommendations field exists for compatibility
                if "model_recommendations" not in parsed_json:
                    parsed_json["model_recommendations"] = ""
                st.warning("Using unvalidated LLM response due to validation errors.")
                cache_result(dataset_hash_query, prompt, parsed_json)
                return parsed_json

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from LLM for combined analysis: {e}. Response: {content[:500]}")
        # Fallback or error handling
        return {
            "insights": "Error: LLM response was not valid JSON.",
            "preprocessing_suggestions": {},
            "visualization_suggestions": [],
            "model_recommendations": ""
        }
    except Exception as e:
        st.error(f"Error in CombinedAnalysisAgent: {e}")
        # Fallback or error handling
        return {
            "insights": f"Error generating insights: {e}",
            "preprocessing_suggestions": {},
            "visualization_suggestions": [],
            "model_recommendations": ""
        }

# === Main Streamlit App ===
def main():
    st.set_page_config(layout="wide", page_title="AskurData Education")
    
    # Initialize session state variables if they don't exist
    default_session_state = {
        "plots": [],
        "messages": [],
        "df": None,
        "df_processed_history": [], # To store states of df for undo or comparison
        "current_file_name": None,
        "insights": None,
        "preprocessing_suggestions": {},
        "visualization_suggestions": [],
        "model_suggestions": None,
        "initial_analysis_done": False # Flag to track if combined analysis is complete
    }
    for key, value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    left, right = st.columns([3,7])
    
    with left:
        st.header("Data Analysis Agent")
        st.markdown("<medium>Powered by <a href='https://build.nvidia.com/nvidia/llama-3.1-nemotron-ultra-253b-v1'>NVIDIA Llama-3.1-Nemotron-Ultra-253b-v1</a></medium>", unsafe_allow_html=True)
        
        # File Uploader
        uploaded_file = st.file_uploader("Choose CSV or Excel", type=["csv", "xlsx"], key="file_uploader")

        if uploaded_file is not None:
            # Check if it's a new file or the same file (to avoid reprocessing on every rerun)
            if st.session_state.current_file_name != uploaded_file.name:
                st.session_state.df = load_data(uploaded_file)
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.messages = [] # Clear messages for new file
                st.session_state.plots = []    # Clear plots for new file
                st.session_state.df_processed_history = [] # Clear history
                st.session_state.initial_analysis_done = False # Reset flag for new file
                # Clear previous analysis results
                st.session_state.insights = None
                st.session_state.preprocessing_suggestions = {}
                st.session_state.visualization_suggestions = []
                st.session_state.model_suggestions = None
                st.rerun() # Rerun once to update UI with new DF and clear old analysis state
            
            if st.session_state.df is not None:
                # Display dataset info always if df is loaded
                st.markdown(f"**Dataset Info: {st.session_state.current_file_name}**")
                st.markdown(f"Rows: {len(st.session_state.df)}, Columns: {len(st.session_state.df.columns)}")
                with st.expander("Column Names and Types"):
                    # Create a DataFrame for column names and types
                    col_info_df = pd.DataFrame({
                        'Column Name': st.session_state.df.columns,
                        'Data Type': [str(dtype) for dtype in st.session_state.df.dtypes]
                    })
                    st.dataframe(col_info_df, use_container_width=True)

                # Display dataset preview in sidebar (first 5 rows)
                # The requirement mentioned preview in chat, but sidebar is also good for persistent view.
                # Chat preview happens on initial load message.
                with st.expander("Dataset Preview (First 5 Rows)", expanded=False):
                    st.dataframe(st.session_state.df.head())

                # Perform combined analysis only if not already done for the current df
                if not st.session_state.initial_analysis_done:
                    with st.spinner("Generating initial dataset analysis. This may take a moment..."):
                        analysis_results = CombinedAnalysisAgent(st.session_state.df)
                        
                        # Initialize with defaults in case of partial failure
                        st.session_state.insights = "Insights generation failed or pending."
                        st.session_state.preprocessing_suggestions = {}
                        st.session_state.visualization_suggestions = []
                        st.session_state.model_suggestions = "Model suggestions generation failed or pending."

                        if isinstance(analysis_results, dict):
                            st.session_state.insights = analysis_results.get("insights", st.session_state.insights)
                            st.session_state.preprocessing_suggestions = analysis_results.get("preprocessing_suggestions", st.session_state.preprocessing_suggestions)
                            st.session_state.visualization_suggestions = analysis_results.get("visualization_suggestions", st.session_state.visualization_suggestions)
                            st.session_state.model_suggestions = analysis_results.get("model_recommendations", st.session_state.model_suggestions)
                            
                            # Log to console if any part of the analysis returned an error string from the agent
                            for key, value in analysis_results.items():
                                if isinstance(value, str) and "Error:" in value:
                                    print(f"CombinedAnalysisAgent returned error for '{key}': {value}")
                        else:
                            st.error("Failed to retrieve a valid analysis structure from the agent.")
                            print("CombinedAnalysisAgent did not return a dictionary.")

                        st.session_state.initial_analysis_done = True # Mark analysis as attempted/done

                        initial_chat_messages = []
                        if st.session_state.insights and "failed or pending" not in st.session_state.insights and "Error:" not in st.session_state.insights:
                            initial_chat_messages.append(f"### Dataset Insights\n{st.session_state.insights}")
                        else:
                            initial_chat_messages.append("### Dataset Insights\nCould not retrieve insights at this time.")
                        
                        # Remove model suggestions from automatic display
                        # Users can access them through the Utilities tab
                        
                        # Add combined initial messages to chat if there's anything to show
                        if initial_chat_messages:
                            # Prepend to messages so it appears first after dataset load info
                            st.session_state.messages.insert(0, {
                                "role": "assistant",
                                "content": "\n\n---\n\n".join(initial_chat_messages)
                            })
                        st.rerun() # Rerun to display the new insights in chat and populate tools
                        if not initial_chat_messages:
                            st.error("Failed to retrieve initial analysis from the agent (result was None or empty).")
        else:
            st.info("Upload a CSV or Excel file to begin.")
            # Clear session state if no file is uploaded or file is removed
            if st.session_state.current_file_name is not None:
                st.session_state.current_file_name = None
                st.session_state.df = None
                st.session_state.messages = []
                st.session_state.plots = []
                st.session_state.initial_analysis_done = False
                st.session_state.insights = None
                st.session_state.preprocessing_suggestions = {}
                st.session_state.visualization_suggestions = []
                st.session_state.model_suggestions = None
                st.rerun() # Rerun to clear the UI

        # Tool Dashboard - should populate based on session_state variables filled by CombinedAnalysisAgent
        st.header("Tool Dashboard")
        tab_pre, tab_eda, tab_utils = st.tabs(["Preprocessing", "EDA", "Utilities"])
        with tab_pre:
            with st.expander("💡 AI Suggestions", expanded=True):
                st.subheader("AI Suggested Preprocessing")
                if st.session_state.initial_analysis_done and st.session_state.preprocessing_suggestions:
                    suggestions_to_display = dict(st.session_state.preprocessing_suggestions)
                    explanation = suggestions_to_display.pop("explanation", None)
                    if not suggestions_to_display:
                        st.caption("No specific preprocessing steps suggested by AI.")
                    for i, (key, desc) in enumerate(suggestions_to_display.items()):
                        button_key = f"preprocess_btn_{key.replace(' ', '_')}_{i}"
                        if st.button(desc, key=button_key, help=f"Apply action: {key}"):
                            # Save a copy of the current df for diff
                            st.session_state.df_before_preprocess = st.session_state.df.copy() if st.session_state.df is not None else None
                            query_for_preprocessing = f"Apply AI suggestion: {desc}"
                            st.session_state.messages.append({"role": "user", "content": query_for_preprocessing})
                            st.session_state.last_preprocess_action = key
                            st.rerun()
                    if explanation:
                        st.markdown(f"**AI Explanation:** {explanation}")
                elif st.session_state.df is not None and not st.session_state.initial_analysis_done:
                    st.caption("Suggestions will appear after initial analysis.")
                elif st.session_state.df is None:
                    st.caption("Upload a dataset to see suggestions.")
                else:
                    st.caption("No preprocessing suggestions from AI.")

                # After processing a preprocessing action, show the modified dataset and what changed
                if (
                    hasattr(st.session_state, 'df_before_preprocess') and
                    st.session_state.df_before_preprocess is not None and
                    st.session_state.df is not None and
                    hasattr(st.session_state, 'last_preprocess_action') and
                    st.session_state.last_preprocess_action is not None
                ):
                    old_df = st.session_state.df_before_preprocess
                    new_df = st.session_state.df
                    changed_cols = [col for col in new_df.columns if not old_df[col].equals(new_df[col]) if col in old_df.columns]
                    added_cols = [col for col in new_df.columns if col not in old_df.columns]
                    removed_cols = [col for col in old_df.columns if col not in new_df.columns]
                    st.markdown(f"### 🛠️ Preprocessing Applied: {st.session_state.last_preprocess_action}")
                    st.markdown(f"**Changed columns:** {', '.join(changed_cols) if changed_cols else 'None'}")
                    st.markdown(f"**Added columns:** {', '.join(added_cols) if added_cols else 'None'}")
                    st.markdown(f"**Removed columns:** {', '.join(removed_cols) if removed_cols else 'None'}")
                    # Show missing value summary before/after
                    old_missing = old_df.isnull().sum().sum()
                    new_missing = new_df.isnull().sum().sum()
                    st.markdown(f"**Missing values before:** {old_missing}, **after:** {new_missing}")
                    st.markdown("#### Preview of Modified Dataset:")
                    st.dataframe(new_df.head())
                    # Reset so this only shows once per action
                    st.session_state.df_before_preprocess = None
                    st.session_state.last_preprocess_action = None

            with st.expander("🛠️ Preprocessing Tools", expanded=True):
                if st.session_state.df is None:
                    st.caption("Upload a dataset to use preprocessing tools.")
                else:
                    # Handle Missing Values
                    st.subheader("Handle Missing Values")
                    with st.form("impute_form"):
                        st.selectbox("Column to Impute", options=st.session_state.df.columns, key="impute_col_select")
                        st.selectbox("Imputation Strategy", options=["mean", "median", "mode", "constant", "forward_fill", "backward_fill"], key="impute_strategy_select")
                        st.text_input("Constant Value (if strategy is 'constant')", key="impute_constant_val")
                        impute_submit = st.form_submit_button("Apply Imputation")
                        if impute_submit:
                            col = st.session_state.impute_col_select
                            strategy = st.session_state.impute_strategy_select
                            const_val = st.session_state.impute_constant_val
                            query = f"Impute column '{col}' with {strategy}"
                            if strategy == "constant":
                                query += f" (value: {const_val})"
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()

                    # Encode Categorical Variables
                    st.subheader("Encode Categorical Variables")
                    with st.form("encode_form"):
                        st.selectbox("Column to Encode", options=st.session_state.df.select_dtypes(include='object').columns, key="encode_col_select")
                        st.selectbox("Encoding Strategy", options=["label_encoding", "one_hot_encoding"], key="encode_strategy_select")
                        encode_submit = st.form_submit_button("Apply Encoding")
                        if encode_submit:
                            col = st.session_state.encode_col_select
                            strategy = st.session_state.encode_strategy_select
                            query = f"{strategy} for column '{col}'"
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
                    
                    # Scale Numerical Features
                    st.subheader("Scale Numerical Features")
                    with st.form("scale_form"):
                        st.multiselect("Columns to Scale", options=st.session_state.df.select_dtypes(include=np.number).columns, key="scale_cols_select")
                        st.selectbox("Scaling Strategy", options=["standard_scaling", "min_max_scaling", "robust_scaling"], key="scale_strategy_select")
                        scale_submit = st.form_submit_button("Apply Scaling")
                        if scale_submit:
                            cols = st.session_state.scale_cols_select
                            strategy = st.session_state.scale_strategy_select
                            if cols:
                                query = f"{strategy} for columns: {', '.join(cols)}"
                                st.session_state.messages.append({"role": "user", "content": query})
                                st.rerun()
                            else:
                                st.warning("Please select columns to scale.")

                    # Outlier Handling
                    st.subheader("Outlier Handling (IQR)")
                    with st.form("outlier_form"):
                        outlier_cols = st.multiselect("Select columns for outlier handling", options=st.session_state.df.select_dtypes(include='number').columns, key="outlier_cols_select")
                        outlier_strategy = st.selectbox("Outlier Strategy", options=["remove", "cap"], key="outlier_strategy_select")
                        outlier_submit = st.form_submit_button("Apply Outlier Handling")
                        if outlier_submit and outlier_cols:
                            query = f"Apply {outlier_strategy} outlier handling to columns: {', '.join(outlier_cols)}"
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
                    
                    # Feature Engineering
                    st.subheader("Feature Engineering")
                    with st.form("feature_eng_form"):
                        feat_type = st.selectbox("Feature Type", ["Polynomial Features", "Date Component Extraction"], key="feat_type_select")
                        if feat_type == "Polynomial Features":
                            poly_cols = st.multiselect("Columns for Polynomial Features", options=st.session_state.df.select_dtypes(include='number').columns, key="poly_cols_select")
                            poly_degree = st.number_input("Polynomial Degree", min_value=2, max_value=5, value=2, key="poly_degree_input")
                        else:
                            date_cols = st.multiselect("Date Columns", options=st.session_state.df.columns, key="date_cols_select")
                        feat_submit = st.form_submit_button("Apply Feature Engineering")
                        if feat_submit:
                            if feat_type == "Polynomial Features" and poly_cols:
                                query = f"Add polynomial features (degree {poly_degree}) for columns: {', '.join(poly_cols)}"
                            elif feat_type == "Date Component Extraction" and date_cols:
                                query = f"Extract date components (year, month, day) from columns: {', '.join(date_cols)}"
                            else:
                                query = None
                            if query:
                                st.session_state.messages.append({"role": "user", "content": query})
                                st.rerun()
                    
                    # Data Filtering (NEW: in Preprocessing tab)
                    st.subheader("Data Filtering")
                    with st.form("preprocess_filter_form"):
                        filter_col = st.selectbox("Column to Filter", options=st.session_state.df.columns, key="preprocess_filter_col_select")
                        filter_op = st.selectbox("Operator", [">", ">=", "<", "<=", "==", "!="], key="preprocess_filter_op_select")
                        filter_val = st.text_input("Value", key="preprocess_filter_val_input")
                        filter_submit = st.form_submit_button("Apply Filter")
                        if filter_submit and filter_col and filter_op and filter_val:
                            query = f"Filter rows where {filter_col} {filter_op} {filter_val}"
                            # Run the filter immediately and update df
                            code, intent, is_plotly, _ = CodeGenerationAgent(query, st.session_state.df)
                            result_obj = ExecutionAgent(code, st.session_state.df, intent, is_plotly, query)
                            if isinstance(result_obj, pd.DataFrame):
                                st.session_state.df = result_obj
                                st.success(f"Filter applied: {filter_col} {filter_op} {filter_val}. Dataset now has {len(result_obj)} rows.")
                                st.dataframe(result_obj.head())
                            else:
                                st.error(f"Filtering failed: {result_obj}")
                    
                    # Custom Preprocessing
                    st.subheader("Custom Preprocessing")
                    with st.form("custom_preprocess_form"):
                        custom_query = st.text_area(
                            "Enter custom preprocessing command:",
                            placeholder="Examples:\n- log transform column Revenue\n- drop column unnecessary_col\n- rename column old_name to new_name\n- bin column age into 5 bins\n- sqrt column price",
                            height=100,
                            key="custom_preprocess_input"
                        )
                        custom_submit = st.form_submit_button("Apply Custom Preprocessing")
                        if custom_submit and custom_query.strip():
                            query = f"Custom preprocess: {custom_query.strip()}"
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
                        elif custom_submit and not custom_query.strip():
                            st.warning("Please enter a custom preprocessing command.")
                    
                    st.caption("💡 **Custom Commands:** log transform, drop column, rename column, bin column, sqrt column")
        with tab_eda:
            # --- AI Suggested EDA (NEW) ---
            with st.expander("💡 AI Suggested EDA", expanded=True):
                st.subheader("AI Suggested Visualizations")
                if st.session_state.initial_analysis_done and st.session_state.visualization_suggestions:
                    for i, suggestion in enumerate(st.session_state.visualization_suggestions):
                        query = suggestion["query"] if isinstance(suggestion, dict) else suggestion.get("query", "")
                        desc = suggestion["desc"] if isinstance(suggestion, dict) else suggestion.get("desc", "")
                        button_key = f"eda_viz_btn_{i}"
                        if st.button(desc, key=button_key, help=f"Show: {query}"):
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
                elif st.session_state.df is not None and not st.session_state.initial_analysis_done:
                    st.caption("Suggestions will appear after initial analysis.")
                elif st.session_state.df is None:
                    st.caption("Upload a dataset to see suggestions.")
                else:
                    st.caption("No EDA suggestions from AI.")
            # --- End AI Suggested EDA ---
            
            with st.expander("📊 Manual Visualizations", expanded=True):
                if st.session_state.df is None:
                    st.caption("Upload a dataset to use EDA tools.")
                else:
                    # Manual Visualizations
                    st.subheader("Create Custom Visualizations")
                    with st.form("manual_viz_form"):
                        plot_type = st.selectbox("Plot Type", ["bar", "line", "pie", "histogram", "scatter", "boxplot", "heatmap"], key="plot_type_select")
                        x_col = st.selectbox("X Axis", options=st.session_state.df.columns, key="x_col_select")
                        y_col = None
                        if plot_type in ["bar", "line", "scatter", "boxplot"]:
                            y_col = st.selectbox("Y Axis", options=st.session_state.df.columns, key="y_col_select")
                        chart_lib = st.selectbox("Chart Library", ["Matplotlib", "Seaborn", "Plotly"], key="chart_lib_select")
                        viz_submit = st.form_submit_button("Generate Visualization")
                        if viz_submit:
                            if plot_type in ["bar", "line", "scatter"] and y_col:
                                query = f"Show {plot_type} chart of {y_col} vs {x_col} using {chart_lib}"
                            elif plot_type == "boxplot" and y_col:
                                query = f"Show boxplot of {y_col} by {x_col} using {chart_lib}"
                            elif plot_type == "heatmap":
                                query = f"Show correlation heatmap using {chart_lib}"
                            elif plot_type in ["pie", "histogram"]:
                                query = f"Show {plot_type} of {x_col} using {chart_lib}"
                            else:
                                query = None
                            if query:
                                st.session_state.messages.append({"role": "user", "content": query})
                                st.rerun()
            
            with st.expander("📈 Quick Analysis", expanded=True):
                if st.session_state.df is None:
                    st.caption("Upload a dataset to use analysis tools.")
                else:
                    # Statistical Summaries
                    st.subheader("Statistical Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Statistical Summary", key="stat_summary_btn"):
                            st.session_state.messages.append({"role": "user", "content": "Show statistical summary (describe)"})
                            st.rerun()
                    with col2:
                        if st.button("Missing Values Analysis", key="missing_vals_analysis_btn"):
                            st.session_state.messages.append({"role": "user", "content": "Show detailed missing value summary and analysis"})
                            st.rerun()
                    
                    # Correlation Analysis
                    st.subheader("Correlation Analysis")
                    if st.button("Show Correlation Matrix Heatmap", key="corr_matrix_btn"):
                        st.session_state.messages.append({"role": "user", "content": "Show correlation matrix heatmap"})
                        st.rerun()
        
        with tab_utils:
            st.subheader("🤖 Model Recommendations")
            if st.session_state.df is None:
                st.caption("Upload a dataset to see model recommendations.")
            else:
                # Model Recommendations Section
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Get AI Model Recommendations", key="model_recom_btn_utils", disabled=not st.session_state.initial_analysis_done):
                        if st.session_state.model_suggestions:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"### AI Model Recommendations\n{st.session_state.model_suggestions}"
                            })
                            st.rerun()
                        else:
                            st.warning("Model suggestions not available from initial analysis.")
                with col2:
                    if st.button("Custom Model Analysis", key="custom_model_btn"):
                        st.session_state.messages.append({"role": "user", "content": "Recommend machine learning models for this dataset"})
                        st.rerun()
            
            st.subheader("📥 Data Export")
            if st.session_state.df is not None:
                csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Current Dataset (CSV)",
                    data=csv_data,
                    file_name=f"processed_{st.session_state.current_file_name if st.session_state.current_file_name else 'dataset.csv'}",
                    mime="text/csv",
                    key="download_csv_utils_btn"
                )
            else:
                st.caption("Upload a dataset to enable download.")

            st.subheader("📋 Chat History Export")
            if st.session_state.messages:
                col1, col2 = st.columns(2)
                with col1:
                    chat_json = json.dumps(st.session_state.messages, indent=2, default=str)
                    st.download_button(
                        label="Download as JSON",
                        data=chat_json,
                        file_name="chat_history.json",
                        mime="application/json",
                        key="download_json_utils_btn"
                    )
                with col2:
                    chat_txt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    st.download_button(
                        label="Download as Text",
                        data=chat_txt,
                        file_name="chat_history.txt",
                        mime="text/plain",
                        key="download_txt_utils_btn"
                    )
            else:
                st.caption("No chat history to export.")
            
            st.subheader("🚀 Performance & Cache")
            cache_stats = get_preprocessing_cache_stats()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Valid Cache Entries", cache_stats["valid_entries"])
            with col2:
                st.metric("Total Cache Entries", cache_stats["total_entries"])
            with col3:
                if cache_stats["expired_entries"] > 0:
                    st.metric("Expired Entries", cache_stats["expired_entries"])
                else:
                    st.metric("Expired Entries", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Refresh Cache Stats", key="refresh_cache_stats_utils_btn"):
                    st.rerun()
            with col2:
                if st.button("Clear Expired Cache", key="clear_cache_utils_btn"):
                    try:
                        conn = init_preprocessing_cache()  # This automatically cleans expired entries
                        conn.close()
                        st.success("Expired cache entries cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing cache: {e}")

    with right:
        st.header("Chat with your data")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        chat_container = st.container()
        chat_render_error = None
        try:
            # === FIX: Process any unprocessed user message from sidebar tools automatically ===
            # Find the latest user message that is not immediately followed by an assistant message
            unprocessed_user_idx = None
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if st.session_state.messages[i]["role"] == "user":
                    if i == len(st.session_state.messages) - 1 or st.session_state.messages[i+1]["role"] != "assistant":
                        unprocessed_user_idx = i
                        break
            if unprocessed_user_idx is not None:
                user_q = st.session_state.messages[unprocessed_user_idx]["content"]
                try:
                    with st.spinner("Working …"):
                        # Add current df to history before any modification by user query
                        if st.session_state.df is not None:
                            st.session_state.df_processed_history.append(st.session_state.df.copy())
                            if len(st.session_state.df_processed_history) > 5: # Keep last 5 states
                                st.session_state.df_processed_history.pop(0)

                        # Specific handling for "Show detailed missing value summary and analysis" from sidebar
                        if user_q.lower() == "show detailed missing value summary and analysis":
                            result = MissingValueAgent(st.session_state.df) # This provides an LLM summary
                            # Calculate detailed missing values
                            missing_values = st.session_state.df.isnull().sum()
                            missing_percent = (missing_values / len(st.session_state.df)) * 100
                            missing_df = pd.DataFrame({
                                'Column': st.session_state.df.columns,
                                'Missing Values': missing_values,
                                'Percentage Missing (%)': missing_percent
                            })
                            missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percentage Missing (%)', ascending=False)
                            st.session_state.messages.insert(unprocessed_user_idx+1, {
                                "role": "assistant",
                                "content": f"### Detailed Missing Value Analysis {result}", # LLM summary
                                "dataframe": missing_df if not missing_df.empty else "No missing values found."
                            })
                        else:
                            code, intent, is_plotly, _ = CodeGenerationAgent(user_q, st.session_state.df)
                            assistant_msg_content = {}
                            
                            # Get any visualization optimization metadata from session state
                            viz_metadata = getattr(st.session_state, '_last_viz_metadata', None)
                            df_viz = getattr(st.session_state, '_last_df_viz', None)
                            
                            if code:
                                result_obj = ExecutionAgent(code, st.session_state.df, intent, is_plotly, user_q, df_viz)
                                if intent == "preprocessing" and isinstance(result_obj, pd.DataFrame):
                                    st.session_state.df = result_obj
                                    assistant_msg_content["dataframe_preview"] = result_obj.head()
                                    assistant_msg_content["header_text"] = "Preprocessing applied successfully! Dataset updated."
                                elif isinstance(result_obj, str) and result_obj.startswith("Error"):
                                    assistant_msg_content["header_text"] = "⚠️ Error"
                                    assistant_msg_content["error_details"] = result_obj
                                elif is_plotly and hasattr(result_obj, '__module__') and result_obj.__module__ and 'plotly' in result_obj.__module__:
                                    st.session_state.plots.append(result_obj)
                                    assistant_msg_content["plot_index"] = len(st.session_state.plots) - 1
                                    assistant_msg_content["is_plotly"] = True
                                    header_text = "📊 Here is the Plotly visualization:"
                                    if viz_metadata and viz_metadata.get("operation") != "none":
                                        header_text += f" (Optimized: {viz_metadata['final_rows']:,} rows)"
                                    assistant_msg_content["header_text"] = header_text
                                elif intent in ["visualization", "chartjs"] and isinstance(result_obj, (plt.Figure, plt.Axes)):
                                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                                    st.session_state.plots.append(fig)
                                    assistant_msg_content["plot_index"] = len(st.session_state.plots) - 1
                                    assistant_msg_content["is_plotly"] = False
                                    header_text = "📊 Here is the visualization:"
                                    if viz_metadata and viz_metadata.get("operation") != "none":
                                        header_text += f" (Optimized: {viz_metadata['final_rows']:,} rows, {viz_metadata['data_reduction']:.1%} reduction)"
                                    assistant_msg_content["header_text"] = header_text
                                elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                                    assistant_msg_content["dataframe_result"] = result_obj
                                    assistant_msg_content["header_text"] = f"🔍 Result:"
                                else:
                                    assistant_msg_content["scalar_result"] = result_obj
                                    assistant_msg_content["header_text"] = f"💡 Result: {str(result_obj)[:200]}"
                                    
                                raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                                reasoning_txt = reasoning_txt.replace("`", "")
                                msg_parts = []
                                if assistant_msg_content.get("header_text"):
                                    msg_parts.append(assistant_msg_content.get("header_text"))
                                
                                # Add library selection information for visualizations
                                if intent in ["visualization"] and hasattr(st.session_state, '_last_library_reason'):
                                    library_info = f"**Library Selection:** {st.session_state._last_library_reason}"
                                    msg_parts.append(library_info)
                                
                                if reasoning_txt:
                                    msg_parts.append(reasoning_txt)
                                thinking_html = f'''<details class="thinking" style="margin-top: 10px; border: 1px solid #ddd; padding: 5px;">\n    <summary>🧠 View Model Reasoning</summary>\n    <pre style="white-space: pre-wrap; word-wrap: break-word;">{raw_thinking}</pre>\n</details>''' if raw_thinking else ""
                                code_html = f'''<details class="code" style="margin-top: 10px; border: 1px solid #ddd; padding: 5px;">\n    <summary>💻 View Generated Code</summary>\n    <pre><code class="language-python" style="white-space: pre-wrap; word-wrap: break-word;">{code}</code></pre>\n</details>'''
                                if thinking_html:
                                    msg_parts.append(thinking_html)
                                if code_html:
                                    msg_parts.append(code_html)
                                final_assistant_text = "\n\n".join(filter(None, msg_parts))
                                df_for_message = None
                                if assistant_msg_content.get("dataframe_preview") is not None:
                                    df_for_message = assistant_msg_content.get("dataframe_preview")
                                elif assistant_msg_content.get("dataframe_result") is not None:
                                    df_for_message = assistant_msg_content.get("dataframe_result")
                                st.session_state.messages.insert(unprocessed_user_idx+1, {
                                    "role": "assistant",
                                    "content": final_assistant_text,
                                    "plot_index": assistant_msg_content.get("plot_index"),
                                    "is_plotly": assistant_msg_content.get("is_plotly", False),
                                    "dataframe": df_for_message,
                                    "chartjs": assistant_msg_content.get("chartjs_config")
                                })
                            else:
                                if "insights" in user_q.lower() or "describe" in user_q.lower() or "dataset contain" in user_q.lower():
                                    insight_text = DataInsightAgent(st.session_state.df)
                                    st.session_state.messages.insert(unprocessed_user_idx+1, {"role": "assistant", "content": f"### Dataset Insights\n{insight_text}"})
                                else:
                                    st.session_state.messages.insert(unprocessed_user_idx+1, {"role": "assistant", "content": "I couldn't generate code or a direct answer for your query. Please try rephrasing or use the sidebar tools."})
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        except Exception as chat_render_exc:
            chat_render_error = chat_render_exc
        # Always render the chat area, even if an error occurred
        with chat_container:
            if chat_render_error:
                st.error(f"A chat processing error occurred: {chat_render_error}")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            plot_obj = st.session_state.plots[idx]
                            is_plotly_fig = msg.get("is_plotly", False)
                            if is_plotly_fig or (hasattr(plot_obj, '__module__') and plot_obj.__module__ and 'plotly' in plot_obj.__module__):
                                st.plotly_chart(plot_obj, use_container_width=True)
                            elif isinstance(plot_obj, (plt.Figure, plt.Axes)):
                                st.pyplot(plot_obj, use_container_width=True)
                            else:
                                st.markdown(f"Debug: Plot object at index {idx} is type {type(plot_obj)}")
                    if msg.get("dataframe") is not None:
                        # Display DataFrame or Series
                        df_to_display = msg["dataframe"]
                        if isinstance(df_to_display, (pd.DataFrame, pd.Series)):
                            st.dataframe(df_to_display)
                        else:
                            st.markdown(f"Debug: Dataframe object is type {type(df_to_display)}")
                    if msg.get("chartjs") is not None:
                        # Legacy Chart.js handling - can be removed as Chart.js is deprecated
                        chart_json_str = json.dumps(msg['chartjs'], indent=2)
                        st.markdown(f"Chart.js Configuration (deprecated):```json{chart_json_str}```")
                        st.caption("Chart.js support has been replaced with Plotly for interactive visualizations.")
            # --- Add chat input at the bottom ---
            user_input = st.chat_input("Type your message and press Enter…")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.rerun()


if __name__ == "__main__":
    main()
