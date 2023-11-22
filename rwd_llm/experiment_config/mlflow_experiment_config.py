import logging
import os
import pprint
from dataclasses import dataclass
import traceback
from typing import Dict, Optional

import mlflow

from ..utils import get_by_path, hydra_utils
from .experiment_config import ExperimentConfig

pp = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)


@dataclass
class MLFlowExperimentConfig(ExperimentConfig):
    # these are used for logging:
    mlflow_tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    params: Optional[dict] = None
    metrics: Optional[dict] = None
    # extra files/directories to log as artifacts
    pre_run_artifacts: Optional[Dict[str, str]] = None
    post_run_artifacts: Optional[Dict[str, str]] = None

    def _log_artifacts(self, artifacts: Optional[Dict[str, str]]):
        if not artifacts:
            return
        for artifact_name, artifact_path in artifacts.items():
            if os.path.isdir(artifact_path):
                logger.info(
                    f"Logging artifact directory {artifact_path} as {artifact_name}"
                )
                mlflow.log_artifacts(artifact_path, artifact_name)
            else:
                logger.info(f"Logging artifact file {artifact_path} as {artifact_name}")
                mlflow.log_artifact(artifact_path, artifact_name)

    def _run(self):
        if self.mlflow_tracking_uri:
            if "//" not in self.mlflow_tracking_uri:
                self.mlflow_tracking_uri = "file://" + self.mlflow_tracking_uri
            logger.info(f"Setting mlflow tracking uri to {self.mlflow_tracking_uri}")
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        experiment_id = None
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id if experiment is not None else None

        with mlflow.start_run(
            experiment_id=experiment_id, run_name=self.run_name
        ) as run:
            logger.info(
                f"MLFlow run_name: {run.info.run_name} run_id: {run.info.run_id}"
            )
            try:
                hydra_config_dir = hydra_utils.get_config_dir()
                mlflow.log_artifacts(hydra_config_dir, artifact_path="configs")
                hydra_overrides = hydra_utils.get_overrides_dict()
                for override_param, override_value in hydra_overrides.items():
                    if override_param.startswith("+"):
                        override_param = override_param[1:]
                    at_loc = override_param.find("@")
                    if at_loc >= 0:
                        override_param = override_param[(at_loc + 1) :]
                    mlflow.log_param(f"hydra_override.{override_param}", override_value)
                hydra_outputs_dir = hydra_utils.get_hydra_out_dir()
                mlflow.log_artifacts(hydra_outputs_dir, artifact_path="hydra")
            except ValueError as err:
                logger.exception(f"Error logging hydra data: {err}")

            logger.info("Logging pre-run artifacts to mlflow")
            self._log_artifacts(self.pre_run_artifacts)

            if self.params is not None:
                mlflow.log_params(self.params)
            logger.info("running experiment")
            results = self.experiment.run()
            if self.metrics is not None:
                metrics = results.eval_metrics
                for metric_name, metric_key in self.metrics.items():
                    metric_val = float("nan")
                    try:
                        metric_val = get_by_path(metrics, metric_key)
                    except Exception as e:
                        logger.error(f"Failed to get metric {metric_key}: {e}")
                    mlflow.log_metric(metric_name, metric_val)
            logger.info("Writing experiment results")
            results.write(self.output_dir)
            logger.info("Logging artifacts to mlflow")
            mlflow.log_artifact(results.get_results_path(self.output_dir))
            mlflow.log_artifact(results.get_eval_path(self.output_dir))
            self._log_artifacts(self.post_run_artifacts)
