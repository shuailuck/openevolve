"""
NPS Binary Classification - Symbolic Regression
Predict telecom customer satisfaction (NPS promoter vs detractor).
func(x, params) returns raw logit scores.
sigmoid(func(x, params)) = P(satisfied=1).

Feature mapping (columns of x):
  x[:,  0]: data_usage_high_frequency_low_saturation_high_saturation__trend1  (1-month trend of data_usage_high_frequency_low_saturation_high_saturation)
  x[:,  1]: long_term_data_silence__vol  (volatility of long_term_data_silence)
  x[:,  2]: arpu  (arpu, current month)
  x[:,  3]: network_quality_complaint_count__trend5  (5-month trend of network_quality_complaint_count)
  x[:,  4]: is_changed_from_non_delegated_to_delegated_payment  (is_changed_from_non_delegated_to_delegated_payment, current month)
  x[:,  5]: flag_act_user__trend5  (5-month trend of flag_act_user)
  x[:,  6]: cross_network_outgoing_call_ratio  (cross_network_outgoing_call_ratio, current month)
  x[:,  7]: arpu__trend5  (5-month trend of arpu)
  x[:,  8]: mnp_port_out_risk__trend5  (5-month trend of mnp_port_out_risk)
  x[:,  9]: card2_dou_hb  (card2_dou_hb, current month)
  x[:, 10]: is_sub_card__trend5  (5-month trend of is_sub_card)
  x[:, 11]: primary_tariff_discount_rate__trend5  (5-month trend of primary_tariff_discount_rate)
  x[:, 12]: family_cnt__vol  (volatility of family_cnt)
  x[:, 13]: churn_risk  (churn_risk, current month)
  x[:, 14]: last_month_bill__trend1  (1-month trend of last_month_bill)
  x[:, 15]: zztd_cnt  (zztd_cnt, current month)
  x[:, 16]: flag_64__trend1  (1-month trend of flag_64)
  x[:, 17]: credit_class_name  (credit_class_name, current month)
  x[:, 18]: card2_call_num_hb__trend1  (1-month trend of card2_call_num_hb)
  x[:, 19]: suspected_illegal_outbound_call_reach__vol  (volatility of suspected_illegal_outbound_call_reach)

Notes:
  - Features with __trend5 suffix: current_value - value_5_months_ago (positive = increasing)
  - Features with __trend1 suffix: current_value - value_1_month_ago (positive = increasing)
  - Features with __vol suffix: std deviation across 6 monthly observations (higher = more volatile)
  - All continuous features are approximately standardized
  - params are optimized externally via L-BFGS-B on binary cross-entropy
"""
import numpy as np

NUM_PARAMS = 15

# EVOLVE-BLOCK-START

def func(x, params):
    """
    Compute raw logit scores for NPS satisfaction prediction.

    Args:
        x: np.ndarray, shape (n_samples, 20) - selected features
        params: np.ndarray, shape (15,) - optimizable parameters

    Returns:
        np.ndarray, shape (n_samples,) - raw logit scores
    """
    logit = params[14]  # bias
    for i in range(min(14, x.shape[1])):
        logit = logit + params[i] * x[:, i]
    return logit

# EVOLVE-BLOCK-END


def run_search():
    return func
