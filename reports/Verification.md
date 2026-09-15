# Execution and verification

## Executed experiment

The pinned Python 3.13 environment was installed locally. The real UNSW-NB15 CSV partitions and source dictionary were downloaded, hashed and validated. All three required model families were tuned with three-fold CV, refit on full development training data, persisted, evaluated and explained with SHAP. The default complete reproduction command is `python -m src.train`.

## Checks

- Pytest covers deterministic safe feature generation, no input mutation, unseen categories, preservation of training imputation statistics, label/ID exclusion, conflicting duplicate handling, saved-model prediction round trips, required feature validation, negative inputs, undefined subgroup rates and explicit infeasible-threshold fallback.
- The Streamlit AppTest loads the actual model and exercises the example-prediction button. This is a programmatic UI interaction check, not a claim of manual browser inspection or public deployment.
- Six notebooks execute with the project's interpreter. They inspect the saved experiment and run assertions without rerunning hyperparameter search. The initial run used Jupyter kernels; the final run uses the `--in-process` IPython runner for environments without subprocess/socket access. Both execute real cell code; each notebook receives a fresh namespace. See `notebook_execution.json`.
- `scripts/verify_project.py` checks input hashes, disjoint predictor signatures, the saved predictor schema, independent recalculation of final metrics, the data dictionary, notebook execution counts/errors, presentation content coverage, README local links, PNG integrity and pytest results. Exact counts are in `artifact_verification.json`.
- Representative EDA, model-comparison and SHAP figures were visually inspected. The model-comparison palette was corrected so precision and FPR have distinct colors.

## Environment and recovery notes

The restricted Windows sandbox blocked external dataset access, virtual-environment temporary files, Random Forest worker pipes and pytest temporary-directory access. The work succeeded using the approved local execution context. An early training run finished modeling/SHAP but reached report generation before that module existed; the completed pipeline was rerun end to end. These failures did not cause substitution of synthetic data or invented outputs.

Notebook execution may emit a Windows asyncio/ZeroMQ compatibility warning while successfully using its selector support. It is not suppressed. The dependency lock captures the exact installed packages; hardware timing and parallel floating-point arithmetic can vary across systems.

The full eight-test suite passed. After replacing the demo chart with the generated SHAP image, the targeted AppTest passed again without the earlier Altair deprecation warning. Its sandboxed process emitted a temporary-directory cleanup PermissionError at interpreter shutdown despite exit code 0; prediction assertions themselves completed successfully.

## Not performed

No live traffic capture, automatic blocking, public app deployment, demographic fairness assessment, independent human review, or external/temporal generalization test was performed. Presentation deliverables are the requested Markdown slide content, not PowerPoint files. An earlier GitHub fetch was blocked by the approval service usage limit. A subsequent authenticated fetch succeeded. The project preserves the original main commit `40b4c0f3cdf39944369ff09f1359222a9909faac`; publication uses a normal push without rewriting history.
