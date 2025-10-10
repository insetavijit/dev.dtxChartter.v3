"""
Enhanced Enterprise Technical Analysis Framework - Refactored
Clean architecture with separated concerns, proper caching, and optimized performance
"""

import numpy as np
import pandas as pd
import talib
from typing import Union, Optional, Dict, List, Callable, Any, Tuple, Protocol
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
from enum import Enum
import re
from functools import lru_cache, wraps
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from contextlib import contextmanager
import hashlib
import time
import tracemalloc
import gc

# Third-party imports for enhanced caching
try:
    from cachetools import TTLCache, LRUCache
except ImportError:
    # Fallback to basic dict if cachetools not available
    TTLCache = dict
    LRUCache = dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

# %%
@dataclass(frozen=True)
class TechnicalAnalysisConfig:
    """Configuration constants for the framework"""
    # Data validation
    MAX_NAN_RATIO: float = 0.5
    MIN_DATA_POINTS: int = 50

    # Caching
    CACHE_SIZE: int = 1000
    CACHE_TTL: int = 3600  # 1 hour

    # Performance
    MAX_WORKERS: int = 4
    CHUNK_SIZE: int = 10000

    # Bollinger Bands
    DEFAULT_BB_PERIOD: int = 20
    DEFAULT_BB_STD: float = 2.0

    # Memory optimization
    OPTIMIZE_MEMORY: bool = True
    KEEP_OHLCV_FLOAT64: bool = True

# %%
class ComparisonType(Enum):
    """Enumeration of supported comparison operations"""
    ABOVE = "above"
    BELOW = "below"
    CROSSED_UP = "crossed_up"
    CROSSED_DOWN = "crossed_down"
    EQUALS = "equals"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"

class IndicatorType(Enum):
    """Enumeration of TALib indicator categories"""
    OVERLAP = "overlap"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    PRICE = "price"
    CYCLE = "cycle"
    PATTERN = "pattern"

# %%
@dataclass
class AnalysisResult:
    """Enhanced data class to encapsulate analysis results"""
    column_name: str
    operation: str
    success: bool
    message: str = ""
    data: Optional[pd.Series] = None
    execution_time: float = 0.0
    memory_usage: int = 0

@dataclass(frozen=True)  # Make it hashable for caching
class IndicatorConfig:
    """Immutable configuration for technical indicators"""
    name: str
    period: Optional[int] = None
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    signal_period: Optional[int] = None
    source_column: str = "Close"
    parameters: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Convert dict to tuple of tuples for hashability
        if hasattr(self.parameters, 'items'):
            object.__setattr__(self, 'parameters', tuple(self.parameters.items()))

class TAException(Exception):
    """Enhanced exception for Technical Analysis operations"""
    def __init__(self, message: str, error_code: str = "GENERAL", details: Optional[Dict] = None):
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

# %%
class PerformanceProfiler:
    """Performance monitoring and optimization utilities"""

    @staticmethod
    def profile_execution(func: Callable) -> Callable:
        """Decorator to profile function execution time and memory"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.perf_counter() - start_time
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                logger.debug(f"{func.__name__} executed in {execution_time:.4f}s, "
                           f"memory: {current / 1024 / 1024:.2f}MB")

        return wrapper

    @staticmethod
    @contextmanager
    def memory_efficient_processing():
        """Context manager for memory-efficient processing"""
        gc.collect()
        try:
            yield
        finally:
            gc.collect()

# %%
class CacheManager:
    """Centralized cache management with configurable strategies"""

    def __init__(self, config: TechnicalAnalysisConfig):
        self.config = config
        if TTLCache != dict:
            self._cache = TTLCache(maxsize=config.CACHE_SIZE, ttl=config.CACHE_TTL)
        else:
            self._cache = {}
            self._max_size = config.CACHE_SIZE

    def get(self, key: str) -> Any:
        """Get item from cache"""
        return self._cache.get(key)

    def put(self, key: str, value: Any) -> None:
        """Put item in cache with size management"""
        if TTLCache == dict and len(self._cache) >= self._max_size:
            # Simple LRU eviction
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = value

    def clear(self) -> None:
        """Clear cache"""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'max_size': getattr(self, '_max_size', self.config.CACHE_SIZE),
            'hit_rate': getattr(self._cache, 'hits', 0) / max(getattr(self._cache, 'misses', 1) + getattr(self._cache, 'hits', 0), 1)
        }

    @staticmethod
    def create_efficient_key(*args) -> str:
        """Create efficient cache key using hashing"""
        key_data = str(args).encode('utf-8')
        return hashlib.md5(key_data).hexdigest()

# %%
class DataValidator:
    """Enhanced data validation with optimized performance"""

    def __init__(self, config: TechnicalAnalysisConfig):
        self.config = config

    @lru_cache(maxsize=128)
    def validate_column_exists(self, columns_hash: int, column: str) -> bool:
        """Cached column existence validation"""
        # Note: This requires pre-computed hash of columns
        return True  # Actual validation done in calling code

    def validate_numeric_column(self, df: pd.DataFrame, column: str) -> bool:
        """Enhanced numeric column validation"""
        if column not in df.columns:
            available = list(df.columns)
            raise TAException(
                f"Column '{column}' not found",
                "COLUMN_NOT_FOUND",
                {"available_columns": available}
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TAException(
                f"Column '{column}' must be numeric, got {df[column].dtype}",
                "INVALID_DTYPE",
                {"column_dtype": str(df[column].dtype)}
            )

        # Check for excessive NaN values
        nan_ratio = df[column].isna().sum() / len(df)
        if nan_ratio > self.config.MAX_NAN_RATIO:
            logger.warning(f"Column '{column}' has {nan_ratio:.1%} NaN values")

        return True

    def validate_ohlcv_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate OHLCV data structure"""
        required_cols = ['Open', 'High', 'Low', 'Close']
        optional_cols = ['Volume']

        validation_results = {}

        for col in required_cols:
            try:
                self.validate_numeric_column(df, col)
                validation_results[col] = True
            except TAException:
                validation_results[col] = False
                logger.warning(f"Required OHLCV column '{col}' is invalid or missing")

        for col in optional_cols:
            if col in df.columns:
                try:
                    self.validate_numeric_column(df, col)
                    validation_results[col] = True
                except TAException:
                    validation_results[col] = False

        return validation_results

    def validate_minimum_data_points(self, df: pd.DataFrame) -> bool:
        """Validate minimum data points for analysis"""
        if len(df) < self.config.MIN_DATA_POINTS:
            raise TAException(
                f"Insufficient data points: {len(df)} < {self.config.MIN_DATA_POINTS}",
                "INSUFFICIENT_DATA"
            )
        return True

# %%
class DataManager:
    """Separated data management responsibilities"""

    def __init__(self, config: TechnicalAnalysisConfig):
        self.config = config
        self.validator = DataValidator(config)

    def prepare_dataframe(self, df: pd.DataFrame, validate_ohlcv: bool = True) -> pd.DataFrame:
        """Prepare and optimize DataFrame"""
        if not isinstance(df, pd.DataFrame):
            raise TAException("Input must be a pandas DataFrame", "INVALID_INPUT")

        if df.empty:
            raise TAException("DataFrame cannot be empty", "EMPTY_DATAFRAME")

        # Validate minimum data points
        self.validator.validate_minimum_data_points(df)

        # Validate OHLCV structure
        if validate_ohlcv:
            validation_results = self.validator.validate_ohlcv_data(df)
            logger.info(f"OHLCV validation: {validation_results}")

        # Create optimized copy
        optimized_df = self._optimize_datatypes(df.copy())

        return optimized_df

    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types with OHLCV preservation"""
        if not self.config.OPTIMIZE_MEMORY:
            return df

        # Preserve OHLCV columns as float64 for TALib compatibility
        protected_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] if self.config.KEEP_OHLCV_FLOAT64 else []

        for col in df.select_dtypes(include=['float64']).columns:
            if col in protected_cols:
                continue

            if df[col].dtype == 'float64':
                # Check if values can fit in float32
                col_min, col_max = df[col].min(), df[col].max()
                if (col_min >= np.finfo(np.float32).min and
                    col_max <= np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)

        # Optimize integer columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0 and col_max <= 255:
                df[col] = df[col].astype('uint8')
            elif col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')

        return df

    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get detailed memory usage statistics"""
        memory_usage = df.memory_usage(deep=True)
        return {
            'total_mb': memory_usage.sum() / 1024 / 1024,
            'per_column_mb': {col: usage / 1024 / 1024 for col, usage in memory_usage.items()},
            'shape': df.shape
        }

# %%
class TALibIndicatorEngine:
    """Optimized TALib indicator calculation engine"""

    def __init__(self, config: TechnicalAnalysisConfig, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self._available_indicators = self._get_available_indicators()

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_available_indicators() -> Dict[str, Dict[str, Any]]:
        """Cache available TALib indicators with metadata"""
        indicators = {}

        for func_name in dir(talib):
            if func_name.isupper() and hasattr(talib, func_name):
                func = getattr(talib, func_name)
                if callable(func):
                    try:
                        info = talib.abstract.Function(func_name).info
                        indicators[func_name] = {
                            'function': func,
                            'info': info,
                            'inputs': info.get('input_names', ['close']),
                            'parameters': info.get('parameters', {}),
                            'outputs': info.get('output_names', [func_name.lower()])
                        }
                    except:
                        # Fallback for indicators without abstract info
                        indicators[func_name] = {
                            'function': func,
                            'info': {},
                            'inputs': ['close'],
                            'parameters': {},
                            'outputs': [func_name.lower()]
                        }

        logger.info(f"Loaded {len(indicators)} TALib indicators")
        return indicators

    def is_indicator_available(self, indicator: str) -> bool:
        """Check if indicator is available"""
        return indicator.upper() in self._available_indicators

    def get_indicator_info(self, indicator: str) -> Dict[str, Any]:
        """Get detailed indicator information"""
        return self._available_indicators.get(indicator.upper(), {})

    @PerformanceProfiler.profile_execution
    def calculate_indicator(self, df: pd.DataFrame, config: IndicatorConfig) -> pd.Series:
        """Calculate technical indicator with optimized caching"""
        indicator_name = config.name.upper()

        if not self.is_indicator_available(indicator_name):
            available = list(self._available_indicators.keys())[:10]  # Show first 10
            raise TAException(
                f"Indicator '{indicator_name}' not available in TALib",
                "INDICATOR_NOT_FOUND",
                {"available_sample": available}
            )

        # Create efficient cache key
        cache_key = self._create_cache_key(df, config)

        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {indicator_name}")
            return cached_result

        try:
            result = self._calculate_indicator_internal(df, config)

            # Cache the result
            self.cache.put(cache_key, result)

            return result

        except Exception as e:
            logger.debug(f"TALib calculation error for {indicator_name}: {str(e)}")
            raise TAException(
                f"Failed to calculate {indicator_name}: {str(e)}",
                "CALCULATION_ERROR",
                {"indicator": indicator_name, "config": config}
            )

    def _create_cache_key(self, df: pd.DataFrame, config: IndicatorConfig) -> str:
        """Create efficient cache key using data fingerprint"""
        source_col = df[config.source_column]

        # Create data fingerprint instead of hashing entire series
        data_fingerprint = (
            len(df),
            source_col.dtype,
            float(source_col.iloc[0]) if len(source_col) > 0 else 0,
            float(source_col.iloc[-1]) if len(source_col) > 0 else 0,
            float(source_col.mean()) if len(source_col) > 0 else 0
        )

        # Create config fingerprint
        config_data = (
            config.name, config.period, config.fast_period,
            config.slow_period, config.signal_period, config.source_column
        )

        return CacheManager.create_efficient_key(data_fingerprint, config_data)

    def _calculate_indicator_internal(self, df: pd.DataFrame, config: IndicatorConfig) -> pd.Series:
        """Internal indicator calculation with proper error handling"""
        indicator_name = config.name.upper()
        func_info = self._available_indicators[indicator_name]
        func = func_info['function']

        # Prepare parameters
        kwargs = self._prepare_parameters(config, func_info)

        # Get input data with proper types
        input_data = self._prepare_input_data(df, config, func_info)

        # Special handling for multi-input indicators
        result = self._execute_talib_function(func, input_data, kwargs, indicator_name)

        # Handle multiple output indicators
        if isinstance(result, tuple):
            result = self._handle_multiple_outputs(result, indicator_name)

        return pd.Series(result, index=df.index, name=f"{config.name}_{config.period}" if config.period else config.name)

    def _prepare_parameters(self, config: IndicatorConfig, func_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for TALib function"""
        kwargs = {}

        # Map standard parameters
        if config.period is not None:
            kwargs['timeperiod'] = config.period
        if config.fast_period is not None:
            kwargs['fastperiod'] = config.fast_period
        if config.slow_period is not None:
            kwargs['slowperiod'] = config.slow_period
        if config.signal_period is not None:
            kwargs['signalperiod'] = config.signal_period

        # Add custom parameters
        if config.parameters:
            kwargs.update(dict(config.parameters))

        return kwargs

    def _prepare_input_data(self, df: pd.DataFrame, config: IndicatorConfig, func_info: Dict[str, Any]) -> List[np.ndarray]:
        """Prepare input data ensuring float64 for TALib compatibility"""
        inputs = func_info.get('inputs', ['close'])
        input_data = []

        column_mapping = {
            'close': config.source_column,
            'real': config.source_column,
            'high': 'High',
            'low': 'Low',
            'open': 'Open',
            'volume': 'Volume'
        }

        for inp in inputs:
            inp_lower = inp.lower()
            target_column = column_mapping.get(inp_lower, config.source_column)

            if target_column in df.columns:
                data = df[target_column].astype(np.float64).values
            else:
                # Fallback to source column or default values
                if inp_lower == 'volume':
                    data = np.ones(len(df), dtype=np.float64)
                else:
                    data = df[config.source_column].astype(np.float64).values

            input_data.append(data)

        return input_data

    def _execute_talib_function(self, func: Callable, input_data: List[np.ndarray],
                               kwargs: Dict[str, Any], indicator_name: str) -> Union[np.ndarray, Tuple]:
        """Execute TALib function with appropriate input handling"""
        with PerformanceProfiler.memory_efficient_processing():
            if indicator_name in ['ADX', 'ADXR', 'AROON', 'AROONOSC', 'CCI', 'DX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX', 'ULTOSC', 'WILLR']:
                # These indicators require High, Low, Close (and sometimes Volume)
                if len(input_data) >= 3:
                    return func(input_data[0], input_data[1], input_data[2], **kwargs)
                else:
                    # Fallback: use close price for all inputs
                    close_data = input_data[0]
                    return func(close_data, close_data, close_data, **kwargs)
            else:
                # Standard execution based on input count
                if len(input_data) == 1:
                    return func(input_data[0], **kwargs)
                elif len(input_data) == 2:
                    return func(input_data[0], input_data[1], **kwargs)
                elif len(input_data) == 3:
                    return func(input_data[0], input_data[1], input_data[2], **kwargs)
                else:
                    return func(*input_data, **kwargs)

    def _handle_multiple_outputs(self, result: Tuple, indicator_name: str) -> np.ndarray:
        """Handle indicators that return multiple values"""
        if indicator_name == 'MACD' and len(result) >= 1:
            return result[0]  # Return MACD line
        elif indicator_name == 'BBANDS' and len(result) >= 3:
            return result[1]  # Return middle band (SMA)
        elif indicator_name == 'STOCH' and len(result) >= 2:
            return result[0]  # Return %K
        elif indicator_name == 'AROON' and len(result) >= 2:
            return result[0]  # Return Aroon Up
        else:
            return result[0]  # Default to first output

    def clear_cache(self):
        """Clear indicator calculation cache"""
        self.cache.clear()
        logger.info("Indicator engine cache cleared")

# %%
class BaseComparator(ABC):
    """Enhanced abstract base class for comparison operations"""

    def __init__(self, config: TechnicalAnalysisConfig):
        self.config = config

    @abstractmethod
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:
        """Perform the comparison operation"""
        pass

    def _generate_column_name(self, x: str, y: Union[str, float], operation: str) -> str:
        """Generate descriptive column name with length limits"""
        y_str = str(y).replace('.', '_').replace('-', 'neg')
        column_name = f"{x}_{operation}_{y_str}"

        # Limit column name length
        if len(column_name) > 50:
            column_name = column_name[:47] + "..."

        return column_name

    def _validate_inputs(self, df: pd.DataFrame, x: str, y: Union[str, float]):
        """Validate inputs before comparison"""
        if x not in df.columns:
            raise TAException(f"Column '{x}' not found", "COLUMN_NOT_FOUND")

        if not pd.api.types.is_numeric_dtype(df[x]):
            raise TAException(f"Column '{x}' must be numeric", "INVALID_DTYPE")

        if isinstance(y, str) and y not in df.columns:
            raise TAException(f"Column '{y}' not found", "COLUMN_NOT_FOUND")

        if isinstance(y, str) and not pd.api.types.is_numeric_dtype(df[y]):
            raise TAException(f"Column '{y}' must be numeric", "INVALID_DTYPE")

# %%
class AboveComparator(BaseComparator):
    """Optimized above comparison with vectorized operations"""

    @PerformanceProfiler.profile_execution
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:

        self._validate_inputs(df, x, y)

        if isinstance(y, (int, float)):
            comparison_array = df[x].values > y
        else:
            comparison_array = df[x].values > df[y].values

        new_col = new_col or self._generate_column_name(x, y, "above")
        df[new_col] = comparison_array.astype(np.int8)  # Use int8 for memory efficiency

        return df

class BelowComparator(BaseComparator):
    """Optimized below comparison"""

    @PerformanceProfiler.profile_execution
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:

        self._validate_inputs(df, x, y)

        if isinstance(y, (int, float)):
            comparison_array = df[x].values < y
        else:
            comparison_array = df[x].values < df[y].values

        new_col = new_col or self._generate_column_name(x, y, "below")
        df[new_col] = comparison_array.astype(np.int8)

        return df

class CrossedUpComparator(BaseComparator):
    """Optimized crossed up detection"""

    @PerformanceProfiler.profile_execution
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:

        self._validate_inputs(df, x, y)

        x_values = df[x].values

        if isinstance(y, (int, float)):
            diff = x_values - y
            diff_prev = np.roll(diff, 1)
        else:
            y_values = df[y].values
            diff = x_values - y_values
            diff_prev = np.roll(diff, 1)

        # Vectorized cross-up detection
        crossed_up = (diff > 0) & (diff_prev <= 0)
        crossed_up[0] = False  # First element cannot be a cross

        new_col = new_col or self._generate_column_name(x, y, "crossed_up")
        df[new_col] = crossed_up.astype(np.int8)

        return df

class CrossedDownComparator(BaseComparator):
    """Optimized crossed down detection"""

    @PerformanceProfiler.profile_execution
    def compare(self, df: pd.DataFrame, x: str, y: Union[str, float],
                new_col: Optional[str] = None) -> pd.DataFrame:

        self._validate_inputs(df, x, y)

        x_values = df[x].values

        if isinstance(y, (int, float)):
            diff = x_values - y
            diff_prev = np.roll(diff, 1)
        else:
            y_values = df[y].values
            diff = x_values - y_values
            diff_prev = np.roll(diff, 1)

        crossed_down = (diff < 0) & (diff_prev >= 0)
        crossed_down[0] = False

        new_col = new_col or self._generate_column_name(x, y, "crossed_down")
        df[new_col] = crossed_down.astype(np.int8)

        return df

# %%
class ComparatorFactory:
    """Enhanced factory with configuration injection"""

    def __init__(self, config: TechnicalAnalysisConfig):
        self.config = config
        self._comparators: Dict[str, BaseComparator] = {
            ComparisonType.ABOVE.value: AboveComparator(config),
            ComparisonType.BELOW.value: BelowComparator(config),
            ComparisonType.CROSSED_UP.value: CrossedUpComparator(config),
            ComparisonType.CROSSED_DOWN.value: CrossedDownComparator(config),
        }

    def get_comparator(self, operation: str) -> BaseComparator:
        """Get comparator instance for the given operation"""
        comparator = self._comparators.get(operation.lower())
        if not comparator:
            available = list(self._comparators.keys())
            raise TAException(
                f"Unsupported operation '{operation}'",
                "UNSUPPORTED_OPERATION",
                {"available_operations": available}
            )
        return comparator

# %%
class QueryParser:
    """Enhanced query parser with better pattern matching"""

    COMPARISON_PATTERNS = {
        r'\babove\b': ComparisonType.ABOVE.value,
        r'\bbelow\b': ComparisonType.BELOW.value,
        r'\bcrossed[\s_]?up\b': ComparisonType.CROSSED_UP.value,
        r'\bcrossed[\s_]?down\b': ComparisonType.CROSSED_DOWN.value,
        r'\bequals?\b': ComparisonType.EQUALS.value,
        r'\bgreater[\s_]?than[\s_]?or[\s_]?equal\b': ComparisonType.GREATER_EQUAL.value,
        r'\bless[\s_]?than[\s_]?or[\s_]?equal\b': ComparisonType.LESS_EQUAL.value,
    }

    @classmethod
    def parse_query(cls, query: str) -> List[Dict[str, Any]]:
        """Enhanced query parsing with better error recovery"""
        operations = []

        for line_num, line in enumerate(query.strip().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                operation = cls._parse_line(line)
                if operation:
                    operation['line_number'] = line_num
                    operations.append(operation)
            except Exception as e:
                logger.warning(f"Error parsing line {line_num}: '{line}' - {e}")
                continue

        return operations

    @classmethod
    def _parse_line(cls, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single query line with better error handling"""
        line_lower = line.lower()

        # Find comparison operation
        comparison = None
        for pattern, comp_type in cls.COMPARISON_PATTERNS.items():
            if re.search(pattern, line_lower):
                comparison = comp_type
                break

        if not comparison:
            logger.warning(f"No valid comparison found in: {line}")
            return None

        # Split by comparison operation
        parts = re.split(r'\b(?:above|below|crossed[\s_]?(?:up|down)|equals?|greater[\s_]?than[\s_]?or[\s_]?equal|less[\s_]?than[\s_]?or[\s_]?equal)\b',
                        line, flags=re.IGNORECASE)

        if len(parts) < 2:
            logger.warning(f"Malformed query line: {line}")
            return None

        column1 = parts[0].strip()
        column2 = parts[1].strip()

        # Try to convert column2 to numeric
        try:
            column2 = float(column2)
        except ValueError:
            pass  # Keep as string

        return {
            'column1': column1,
            'operation': comparison,
            'column2': column2,
            'original_line': line
        }

    @staticmethod
    def extract_indicators(query: str) -> List[IndicatorConfig]:
        """Extract indicator configurations from query with better parsing"""
        indicators = []
        # Find indicator patterns like EMA_21, RSI_14, etc.
        indicator_patterns = re.findall(r'\b([A-Z]+)_(\d+)\b', query.upper())

        for name, period in indicator_patterns:
            if name not in ['ABOVE', 'BELOW', 'CROSSED', 'UP', 'DOWN', 'EQUALS']:
                indicators.append(IndicatorConfig(name=name, period=int(period)))

        # Find standalone indicators
        standalone_indicators = re.findall(r'\b([A-Z]{3,})\b', query.upper())
        for name in standalone_indicators:
            if name not in ['ABOVE', 'BELOW', 'CROSSED', 'DOWN', 'EQUALS'] and len(name) >= 3:
                # Check if it's not already added with a period
                if not any(ind.name == name for ind in indicators):
                    indicators.append(IndicatorConfig(name=name))

        # Remove duplicates while preserving order
        unique_indicators = []
        seen = set()
        for ind in indicators:
            key = (ind.name, ind.period)
            if key not in seen:
                unique_indicators.append(ind)
                seen.add(key)

        return unique_indicators

# %%
class SignalGenerator:
    """Separated signal generation logic"""

    def __init__(self, config: TechnicalAnalysisConfig):
        self.config = config

    def combine_signals(self, df: pd.DataFrame, signal_columns: List[str],
                       output_column: str, operation: str = 'AND') -> pd.DataFrame:
        """Combine multiple signal columns with specified operation"""
        available_signals = [col for col in signal_columns if col in df.columns]

        if not available_signals:
            logger.warning(f"No signal columns found for combination: {signal_columns}")
            return df

        if operation.upper() == 'AND':
            combined = df[available_signals[0]].astype(bool)
            for signal in available_signals[1:]:
                combined = combined & df[signal].astype(bool)
        elif operation.upper() == 'OR':
            combined = df[available_signals[0]].astype(bool)
            for signal in available_signals[1:]:
                combined = combined | df[signal].astype(bool)
        else:
            raise TAException(f"Unsupported signal combination operation: {operation}")

        df[output_column] = combined.astype(np.int8)
        logger.info(f"Combined {len(available_signals)} signals into '{output_column}' using {operation}")

        return df

    def generate_trend_following_signals(self, df: pd.DataFrame,
                                       indicator_engine: TALibIndicatorEngine) -> pd.DataFrame:
        """Generate trend-following signals with robust error handling"""

        # Required indicators
        indicators = [
            IndicatorConfig(name='EMA', period=21),
            IndicatorConfig(name='EMA', period=50),
            IndicatorConfig(name='RSI', period=14),
        ]

        # Add indicators
        for indicator in indicators:
            try:
                column_name = f"{indicator.name}_{indicator.period}"
                if column_name not in df.columns:
                    result = indicator_engine.calculate_indicator(df, indicator)
                    df[column_name] = result
                    logger.info(f"Added indicator: {column_name}")
            except Exception as e:
                logger.warning(f"Could not add {indicator.name}_{indicator.period}: {e}")
                continue

        return df

    def generate_mean_reversion_signals(self, df: pd.DataFrame,
                                      indicator_engine: TALibIndicatorEngine) -> pd.DataFrame:
        """Generate mean reversion signals"""

        try:
            # Add RSI
            rsi_config = IndicatorConfig(name='RSI', period=14)
            if 'RSI_14' not in df.columns:
                df['RSI_14'] = indicator_engine.calculate_indicator(df, rsi_config)

            # Add simple moving average for Bollinger Bands middle
            sma_config = IndicatorConfig(name='SMA', period=self.config.DEFAULT_BB_PERIOD)
            if f'SMA_{self.config.DEFAULT_BB_PERIOD}' not in df.columns:
                df[f'SMA_{self.config.DEFAULT_BB_PERIOD}'] = indicator_engine.calculate_indicator(df, sma_config)

            # Calculate Bollinger Bands manually
            close_prices = df['Close']
            sma_col = f'SMA_{self.config.DEFAULT_BB_PERIOD}'
            if sma_col in df.columns:
                sma = df[sma_col]
                rolling_std = close_prices.rolling(window=self.config.DEFAULT_BB_PERIOD).std()

                df['BB_UPPER'] = sma + (self.config.DEFAULT_BB_STD * rolling_std)
                df['BB_LOWER'] = sma - (self.config.DEFAULT_BB_STD * rolling_std)
                df['BB_MIDDLE'] = sma

                logger.info("Generated Bollinger Bands")

        except Exception as e:
            logger.warning(f"Could not generate mean reversion indicators: {e}")

        return df

# %%
class PerformanceAnalyzer:
    """Separated performance analysis and backtesting"""

    def __init__(self, config: TechnicalAnalysisConfig):
        self.config = config

    def backtest_signals(self, df: pd.DataFrame, signal_column: str,
                        entry_price_column: str = 'Close',
                        holding_period: int = 1) -> Dict[str, float]:
        """Enhanced backtesting with more metrics"""

        if signal_column not in df.columns:
            raise TAException(f"Signal column '{signal_column}' not found")

        if entry_price_column not in df.columns:
            raise TAException(f"Price column '{entry_price_column}' not found")

        signals = df[signal_column]
        prices = df[entry_price_column]

        # Find signal entry points
        entry_points = np.where(signals == 1)[0]

        if len(entry_points) == 0:
            return {
                'total_signals': 0,
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }

        returns = []
        equity_curve = []

        for entry_idx in entry_points:
            exit_idx = min(entry_idx + holding_period, len(prices) - 1)

            if exit_idx > entry_idx:
                entry_price = prices.iloc[entry_idx]
                exit_price = prices.iloc[exit_idx]

                if entry_price != 0:  # Avoid division by zero
                    ret = (exit_price - entry_price) / entry_price
                    returns.append(ret)
                    equity_curve.append(1 + ret if not equity_curve else equity_curve[-1] * (1 + ret))

        if not returns:
            return {
                'total_signals': len(entry_points),
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }

        returns_array = np.array(returns)

        # Calculate metrics
        win_rate = (returns_array > 0).sum() / len(returns_array)
        avg_return = returns_array.mean()
        total_return = returns_array.sum()

        # Calculate max drawdown
        equity_curve = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = (avg_return * 252) / (returns_array.std() * np.sqrt(252)) if returns_array.std() > 0 else 0

        return {
            'total_signals': len(entry_points),
            'total_trades': len(returns),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'best_return': returns_array.max(),
            'worst_return': returns_array.min(),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': returns_array.std()
        }

    def generate_performance_report(self, operations_log: List[AnalysisResult]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""

        total_operations = len(operations_log)
        successful_operations = sum(1 for op in operations_log if op.success)
        total_execution_time = sum(op.execution_time for op in operations_log)

        return {
            'summary': {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'success_rate': (successful_operations / total_operations * 100) if total_operations > 0 else 0,
                'total_execution_time_ms': total_execution_time * 1000,
                'avg_execution_time_ms': (total_execution_time / total_operations * 1000) if total_operations > 0 else 0
            },
            'operation_details': [
                {
                    'operation': op.operation,
                    'success': op.success,
                    'execution_time_ms': op.execution_time * 1000,
                    'message': op.message
                } for op in operations_log
            ],
            'failed_operations': [
                op.operation for op in operations_log if not op.success
            ]
        }

# %%
class EnhancedTechnicalAnalyzer:
    """Main analyzer class with clean architecture and separated concerns"""

    def __init__(self, df: pd.DataFrame,
                 config: Optional[TechnicalAnalysisConfig] = None,
                 validate_ohlcv: bool = True):
        """Initialize with dependency injection and clean architecture"""

        # Configuration
        self.config = config or TechnicalAnalysisConfig()

        # Core components
        self.cache_manager = CacheManager(self.config)
        self.data_manager = DataManager(self.config)
        self.indicator_engine = TALibIndicatorEngine(self.config, self.cache_manager)
        self.comparator_factory = ComparatorFactory(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.performance_analyzer = PerformanceAnalyzer(self.config)

        # Data preparation
        self._original_df = df.copy()
        self._df = self.data_manager.prepare_dataframe(df, validate_ohlcv)
        self._operations_log: List[AnalysisResult] = []

        # Log initialization
        memory_info = self.data_manager.get_memory_usage(self._df)
        logger.info(f"Initialized analyzer - Shape: {self._df.shape}, Memory: {memory_info['total_mb']:.2f}MB")

    @property
    def df(self) -> pd.DataFrame:
        """Get the current DataFrame"""
        return self._df

    @property
    def operations_log(self) -> List[AnalysisResult]:
        """Get log of all operations performed"""
        return self._operations_log

    def add_indicator(self, config: IndicatorConfig,
                     column_name: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        """Add technical indicator with enhanced error handling"""

        start_time = time.perf_counter()

        try:
            result = self.indicator_engine.calculate_indicator(self._df, config)

            # Generate column name
            if column_name is None:
                if config.period:
                    column_name = f"{config.name}_{config.period}"
                else:
                    column_name = config.name

            self._df[column_name] = result
            execution_time = time.perf_counter() - start_time

            # Log successful operation
            analysis_result = AnalysisResult(
                column_name=column_name,
                operation=f"ADD_INDICATOR_{config.name}",
                success=True,
                message="Indicator added successfully",
                data=result,
                execution_time=execution_time
            )
            self._operations_log.append(analysis_result)

            logger.info(f"[OK] Added indicator {column_name} in {execution_time:.4f}s")

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            analysis_result = AnalysisResult(
                column_name=column_name or config.name,
                operation=f"ADD_INDICATOR_{config.name}",
                success=False,
                message=str(e),
                execution_time=execution_time
            )
            self._operations_log.append(analysis_result)
            logger.error(f"[Err] Failed to add indicator {config.name}: {e}")
            raise TAException(f"Failed to add indicator: {e}", "INDICATOR_ADD_FAILED")

        return self

    # Fluent interface methods
    def above(self, x: str, y: Union[str, float], new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        """Fluent interface for above comparison"""
        return self._execute_comparison(ComparisonType.ABOVE.value, x, y, new_col)

    def below(self, x: str, y: Union[str, float], new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        """Fluent interface for below comparison"""
        return self._execute_comparison(ComparisonType.BELOW.value, x, y, new_col)

    def crossed_up(self, x: str, y: Union[str, float], new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        """Fluent interface for crossed up detection"""
        return self._execute_comparison(ComparisonType.CROSSED_UP.value, x, y, new_col)

    def crossed_down(self, x: str, y: Union[str, float], new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        """Fluent interface for crossed down detection"""
        return self._execute_comparison(ComparisonType.CROSSED_DOWN.value, x, y, new_col)

    @PerformanceProfiler.profile_execution
    def _execute_comparison(self, operation: str, x: str, y: Union[str, float],
                          new_col: Optional[str] = None) -> 'EnhancedTechnicalAnalyzer':
        """Execute comparison operation with enhanced error handling"""

        start_time = time.perf_counter()

        try:
            comparator = self.comparator_factory.get_comparator(operation)
            self._df = comparator.compare(self._df, x, y, new_col)

            execution_time = time.perf_counter() - start_time
            result_col = new_col or comparator._generate_column_name(x, y, operation)

            result = AnalysisResult(
                column_name=result_col,
                operation=f"{x} {operation} {y}",
                success=True,
                message="Comparison completed successfully",
                data=self._df[result_col],
                execution_time=execution_time
            )
            self._operations_log.append(result)
            logger.info(f"[ OK ] {result.operation} -> {result.column_name} ({execution_time:.4f}s)")

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            result = AnalysisResult(
                column_name="",
                operation=f"{x} {operation} {y}",
                success=False,
                message=str(e),
                execution_time=execution_time
            )
            self._operations_log.append(result)
            logger.error(f"[Err] {result.operation}: {result.message}")
            raise TAException(f"Comparison operation failed: {e}", "COMPARISON_FAILED")

        return self

    def execute_query(self, query: str, auto_add_indicators: bool = True) -> 'EnhancedTechnicalAnalyzer':
        """Execute natural language query with automatic indicator addition"""

        try:
            # Extract and add indicators if requested
            if auto_add_indicators:
                indicators = QueryParser.extract_indicators(query)
                logger.info(f"Found {len(indicators)} indicators in query")

                for indicator_config in indicators:
                    try:
                        self.add_indicator(indicator_config)
                    except Exception as e:
                        logger.warning(f"Could not add indicator {indicator_config.name}: {e}")

            # Parse and execute comparisons
            operations = QueryParser.parse_query(query)
            logger.info(f"Parsed {len(operations)} operations from query")

            successful_operations = 0
            for op in operations:
                try:
                    self._execute_comparison(op['operation'], op['column1'], op['column2'])
                    successful_operations += 1
                except Exception as e:
                    logger.error(f"Failed to execute query operation {op}: {e}")
                    continue

            logger.info(f"Successfully executed {successful_operations}/{len(operations)} operations")

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise TAException(f"Query execution failed: {e}", "QUERY_EXECUTION_FAILED")

        return self

    def generate_trend_following_signals(self) -> 'EnhancedTechnicalAnalyzer':
        """Generate trend-following signals using signal generator"""
        self._df = self.signal_generator.generate_trend_following_signals(self._df, self.indicator_engine)

        # Create combined signal if components are available
        signal_components = ['Close_above_EMA_21', 'EMA_21_above_EMA_50', 'RSI_14_below_70_0']
        available_signals = [col for col in signal_components if col in self._df.columns]

        if len(available_signals) >= 2:
            self._df = self.signal_generator.combine_signals(
                self._df, available_signals, 'trend_following_signal', 'AND'
            )

        return self

    def generate_mean_reversion_signals(self) -> 'EnhancedTechnicalAnalyzer':
        """Generate mean reversion signals using signal generator"""
        self._df = self.signal_generator.generate_mean_reversion_signals(self._df, self.indicator_engine)

        # Create combined signal if components are available
        signal_components = ['Close_below_BB_LOWER', 'RSI_14_below_30_0']
        available_signals = [col for col in signal_components if col in self._df.columns]

        if len(available_signals) >= 1:
            self._df = self.signal_generator.combine_signals(
                self._df, available_signals, 'mean_reversion_signal', 'AND'
            )

        return self

    def get_signals(self, column: str) -> pd.Series:
        """Get signal series for a specific column"""
        if column not in self._df.columns:
            raise TAException(f"Column '{column}' not found", "COLUMN_NOT_FOUND")
        return self._df[column]

    def get_active_signals(self, column: str, include_index: bool = True) -> pd.DataFrame:
        """Get only rows where signal is active"""
        if column not in self._df.columns:
            raise TAException(f"Column '{column}' not found", "COLUMN_NOT_FOUND")

        active_mask = self._df[column] == 1
        result_df = self._df[active_mask]

        return result_df if include_index else result_df.reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """Enhanced summary with performance metrics"""
        summary_data = []
        for result in self._operations_log:
            summary_data.append({
                'Operation': result.operation,
                'Column': result.column_name,
                'Success': result.success,
                'Execution_Time_ms': round(result.execution_time * 1000, 2),
                'Active_Signals': result.data.sum() if result.data is not None else 0,
                'Signal_Ratio_%': round((result.data.sum() / len(result.data) * 100) if result.data is not None else 0, 2),
                'Message': result.message
            })
        return pd.DataFrame(summary_data)

    def performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        base_report = self.performance_analyzer.generate_performance_report(self._operations_log)

        # Add memory and cache statistics
        memory_info = self.data_manager.get_memory_usage(self._df)
        cache_stats = self.cache_manager.get_stats()

        base_report.update({
            'memory_usage': memory_info,
            'cache_statistics': cache_stats,
            'dataframe_info': {
                'shape': self._df.shape,
                'generated_columns': len([col for col in self._df.columns if col not in self._original_df.columns]),
                'total_columns': len(self._df.columns)
            }
        })

        return base_report

    def backtest_signals(self, signal_column: str, entry_price_column: str = 'Close',
                        holding_period: int = 1) -> Dict[str, float]:
        """Backtest signals using performance analyzer"""
        return self.performance_analyzer.backtest_signals(
            self._df, signal_column, entry_price_column, holding_period
        )

    def reset(self) -> 'EnhancedTechnicalAnalyzer':
        """Reset to original DataFrame state with cache clearing"""
        self._df = self.data_manager.prepare_dataframe(self._original_df)
        self._operations_log.clear()
        self.cache_manager.clear()
        logger.info("Reset analyzer to original state")
        return self

    def export_signals(self, filename: str, format: str = 'csv',
                      columns: Optional[List[str]] = None) -> bool:
        """Export signals with enhanced options"""
        try:
            if columns is None:
                # Auto-detect signal columns
                signal_columns = [col for col in self._df.columns
                                if any(op in col.lower() for op in ['above', 'below', 'crossed', 'signal'])]
                columns = signal_columns + ['Close']

            export_df = self._df[columns].copy()

            if format.lower() == 'csv':
                export_df.to_csv(filename, index=True)
            elif format.lower() == 'parquet':
                export_df.to_parquet(filename, index=True)
            elif format.lower() == 'excel':
                export_df.to_excel(filename, index=True)
            else:
                raise TAException(f"Unsupported export format: {format}", "UNSUPPORTED_FORMAT")

            logger.info(f"Exported {len(columns)} columns to {filename}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

# %%
def create_analyzer(df: pd.DataFrame,
                   config: Optional[TechnicalAnalysisConfig] = None,
                   validate_ohlcv: bool = True) -> EnhancedTechnicalAnalyzer:
    """Factory function to create EnhancedTechnicalAnalyzer instance"""
    return EnhancedTechnicalAnalyzer(df, config, validate_ohlcv)

# Alias for backward compatibility
def cabr(df: pd.DataFrame) -> EnhancedTechnicalAnalyzer:
    """Legacy factory function for compatibility"""
    return EnhancedTechnicalAnalyzer(df)
