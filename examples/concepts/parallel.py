"""
This is a stubbed pipeline that demonstrates parallel
    pipeline execution.
    Note that the branches schema is same as dag schema.

    All the steps are mocked and they will just pass through.
    Use this pattern to define the skeleton of your pipeline and
    flesh out the steps later.

    You can run this pipeline by:
       python examples/concepts/parallel.py
"""

from runnable import Parallel, Pipeline, Stub


def main():
    # The steps in XGBoost training pipeline
    prepare_xgboost = Stub(name="Prepare for XGBoost")
    train_xgboost = Stub(name="Train XGBoost", terminate_with_success=True)

    # prepare_xgboost >> train_xgboost

    # The pipeline for XGBoost training
    xgboost = Pipeline(
        name="XGBoost",
        steps=[prepare_xgboost, train_xgboost],
        # start_at=prepare_xgboost,
        add_terminal_nodes=True,
    )

    # The steps and pipeline  in Random Forest training
    train_rf = Stub(name="Train RF", terminate_with_success=True)
    rfmodel = Pipeline(
        steps=[train_rf],
        # start_at=train_rf,
        add_terminal_nodes=True,
    )

    # The steps in parent pipeline
    get_features = Stub(name="Get Features")
    # The parallel step definition.
    # Branches are just pipelines themselves
    train_models = Parallel(
        name="Train Models",
        branches={"XGBoost": xgboost, "RF Model": rfmodel},
    )
    ensemble_model = Stub(name="Ensemble Modelling")
    run_inference = Stub(name="Run Inference", terminate_with_success=True)

    # get_features >> train_models >> ensemble_model >> run_inference

    # The parent pipeline
    pipeline = Pipeline(
        steps=[get_features, train_models, ensemble_model, run_inference],
        # start_at=get_features,
        add_terminal_nodes=True,
    )

    _ = pipeline.execute(configuration_file="examples/configs/fs-catalog-run_log.yaml")

    return pipeline


if __name__ == "__main__":
    main()
