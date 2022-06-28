FID = 'fid'
FIELD_ID = 'field_id'
TRACE_TIPE = 'trace_type'
IS_XSRC = 'is_xsrc'
IS_YSRC = 'is_ysrc'
IS_ZSRC = 'is_zsrc'
EXISTS = 'exists'
LENGTH = 'length'
DT_IS_STRING = 'data_type_is_string'
DT_IS_INT = 'data_type_is_integer'
DT_IS_DEC = 'data_type_is_decimal'
DT_IS_TIME =  'data_type_is_time'
GT_IS_C = 'general_type_is_c'
GT_IS_Q = 'general_type_is_q'
GT_IS_T = 'general_type_is_t'
HAS_NONE = 'has_none'
PERCENTAGE_NONE = 'percentage_none'
NUM_NONE = 'num_none'
IS_UNIQUE = 'is_unique'
LIST_ENTROPY = 'list_entropy'
HAS_OUTLIERS_15IQR = 'has_outliers_15iqr'
HAS_OUTLIERS_3IQR = 'has_outliers_3iqr'
HAS_OUTLIERS_1_99 = 'has_outliers_1_99'
HAS_OUTLIERS_3STD = 'has_outliers_3std'
IS_NORMAL_5 = 'is_normal_5'
IS_NORMAL_1 = 'is_normal_1'
IS_SORTED = 'is_sorted'
IS_MONOTONIC = 'is_monotonic'
IS_LIN_SPACE = 'is_lin_space'
IS_LOG_SPACE = 'is_log_space'
IS_ONLY_FIELD = 'is_only_field'
NUM_UNIQUE_ELEMS = 'num_unique_elements'
UNIQUE_PERCENT = 'unique_percent'
SORTEDNESS = 'sortedness'
PERCENT_OUTLIERS_15IQR = 'percent_outliers_15iqr'
PERCENT_OUTLIERS_3IQR = 'percent_outliers_3iqr'
PERCENT_OUTLIERS_1_99 = 'percent_outliers_1_99'
PERCENT_OUTLIERS_3STD = 'percent_outliers_3std'
ENTROPY = 'entropy'
GINI = 'gini'
SKEWNESS = 'skewness'
KURTOSIS = 'kurtosis'
MOMENT_5 = 'moment_5'
MOMENT_6 = 'moment_6'
MOMENT_7 = 'moment_7'
MOMENT_8 = 'moment_8'
MOMENT_9 = 'moment_9'
MOMENT_10 = 'moment_10'
COEFF_VAR = 'coeff_var'
QUANT_COEFF_DISP = 'quant_coeff_disp'
LIN_SPACE_SEQ_COEFF = 'lin_space_sequence_coeff'
LOG_SPACE_SEQ_COEFF = 'log_space_sequence_coeff'
NORMAL_STAT = 'normality_statistic'
NORMAL_P = 'normality_p'

identifiers = [
    FID, 
    FIELD_ID
]

outcomes = [
    TRACE_TIPE, 
    IS_XSRC, 
    IS_YSRC, 
    IS_ZSRC
]

general_features = [
    EXISTS, 
    LENGTH, 
    DT_IS_STRING, 
    DT_IS_INT, 
    DT_IS_DEC, 
    DT_IS_TIME, 
    GT_IS_C, 
    GT_IS_Q, 
    GT_IS_T
]

categorical_features = [
    HAS_NONE, 
    PERCENTAGE_NONE, 
    NUM_NONE, IS_UNIQUE, 
    LIST_ENTROPY, 
    HAS_OUTLIERS_15IQR, 
    HAS_OUTLIERS_3IQR, 
    HAS_OUTLIERS_1_99, 
    HAS_OUTLIERS_3STD, 
    IS_NORMAL_1, 
    IS_NORMAL_5, 
    IS_SORTED, 
    IS_MONOTONIC, 
    IS_LIN_SPACE, 
    IS_LOG_SPACE, 
    IS_ONLY_FIELD
]

quantitative_features = [
    NUM_UNIQUE_ELEMS,
    UNIQUE_PERCENT,
    SORTEDNESS,
    PERCENT_OUTLIERS_15IQR,
    PERCENT_OUTLIERS_3IQR,
    PERCENT_OUTLIERS_1_99,
    PERCENT_OUTLIERS_3STD,
    ENTROPY,
    GINI,
    SKEWNESS,
    KURTOSIS,
    MOMENT_5,
    MOMENT_6,
    MOMENT_7,
    MOMENT_8,
    MOMENT_9,
    MOMENT_10,
    COEFF_VAR,
    QUANT_COEFF_DISP,
    LIN_SPACE_SEQ_COEFF,
    LOG_SPACE_SEQ_COEFF,
    NORMAL_STAT,
    NORMAL_P
]

all_columns = identifiers + outcomes + general_features + categorical_features + quantitative_features
outcomes_columns = identifiers + outcomes
features_columns = identifiers + general_features + categorical_features + quantitative_features