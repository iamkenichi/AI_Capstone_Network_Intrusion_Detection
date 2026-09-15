# Explainability

SHAP explains **XGBoost**, the best tree family by validation average precision. The selected deployment prototype model is XGBoost. Global plots use 160 deterministic random primary-test examples; local examples are the first available member of each prediction outcome. Local examples illustrate behavior and are not representative estimates.

## Global behavior
Largest mean absolute contributions: numeric__sttl, numeric__ct_state_ttl, numeric__ct_srv_dst, numeric__dbytes, numeric__ct_dst_src_ltm. `shap_beeswarm.png` shows direction and spread, while `shap_importance.png` shows magnitude. Positive contributions push toward attack and negative contributions toward benign, measured in **log odds** for this explainer. Contributions refer to transformed inputs; numeric colors therefore reflect standardized values, and one-hot colors indicate category activation. Magnitude alone gives no direction; inspect the beeswarm or local signs.

TTL, state and context-count features can reflect capture topology, device defaults or traffic-generator behavior. Dominance of these variables is a reason for transferability testing and feature ablation, not evidence that all high-TTL flows are malicious. `id`, `label` and `attack_cat` are explicitly absent from the model schema. Correlated features can share attribution, and SHAP is neither causal proof nor a security rule.

## Local explanations
### True Positive

Row 244, category Reconnaissance, attack score 0.9759. Strongest contributions: numeric__sttl (+1.874); numeric__ct_srv_dst (+0.871); numeric__ct_srv_src (+0.427); numeric__tcprtt (+0.390); numeric__synack (+0.273); numeric__smean (+0.157).

### True Negative

Row 35, category Normal, attack score 0.3755. Strongest contributions: numeric__sttl (+1.507); numeric__ct_srv_dst (-0.368); numeric__smean (-0.272); numeric__sbytes (-0.267); numeric__synack (-0.158); numeric__dmean (-0.140).

### False Positive

Row 1, category Normal, attack score 0.9048. Strongest contributions: numeric__sttl (+1.663); numeric__ct_srv_dst (-0.322); numeric__ct_srv_src (+0.249); numeric__synack (+0.247); numeric__sbytes (+0.201); numeric__dur (+0.198).

### False Negative

Row 278, category Fuzzers, attack score 0.4269. Strongest contributions: numeric__sttl (+1.459); numeric__ct_srv_dst (-0.944); numeric__ct_dst_src_ltm (-0.187); numeric__sbytes (-0.180); numeric__ct_srv_src (-0.166); categorical__service_dns (-0.078).

Unavailable outcome classes: []. No synthetic example is substituted for an unavailable error class. Waterfall plots are stored as `figures/shap_<outcome>.png`; complete signed contributions for the displayed leading features are in `shap_details.json`.
