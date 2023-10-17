import logging
from typing import Any, Dict

from rwd_llm.chains.chain_utils import ERROR_LABEL
from sklearn.metrics import classification_report

from ..data_loaders import DatasetBase

logger = logging.getLogger(__name__)


def gold_label_eval(gold_labels, predicted_labels):
    """Evaluate the gold labels against the predicted labels.

    Args:
        gold_labels (Dict[str, Any]): Dict of gold labels, keyed by id.
        predicted_labels (Dict[str, Any]): Dict of predicted labels, keyed by id.

    Returns:
        Dict[str, Any]: Dict of evaluation metrics.
    """
    return classification_report(
        y_true=gold_labels, y_pred=predicted_labels, output_dict=True
    )


class Evaluation:
    def evaluate_results(self, dataset: DatasetBase, results: Dict[str, Any]):
        raise NotImplementedError()


class ClassificationEvaluation(Evaluation):
    def __init__(self, label_key: str = "label"):
        self.label_key = label_key

    def evaluate_results(self, dataset: DatasetBase, results: Dict[str, Any]):
        if dataset.labels is None:
            return None
        gold_labels = dataset.labels
        gold_ids = set(dataset.labels.keys())
        predicted_ids = set(results.keys())
        if gold_ids != predicted_ids:
            missing_gold = gold_ids - predicted_ids
            missing_pred = predicted_ids - gold_ids
            logger.warning(
                f"Warning: gold_ids != predicted_ids. Missing gold: {missing_gold},"
                f" Missing predicted: {missing_pred}"
            )
        ids = gold_ids.intersection(predicted_ids)
        gold_labels = {_id: gold_labels[_id].label for _id in ids}
        predicted_labels = {}
        for _id in ids:
            # add gold label to results dict, for debugging FP and FN in results.json
            # output.
            result = results[_id]
            label = None
            gold_label = gold_labels[_id]

            if isinstance(result, Exception):
                label = ERROR_LABEL
            elif isinstance(result, dict):
                # add gold label to result dict, just for convenience
                result["gold_label"] = gold_label
                # find the result label
                if self.label_key not in result:
                    logger.warning(
                        f"Expected label key {self.label_key} in result: {result}"
                    )
                    label = ERROR_LABEL
                else:
                    label = result[self.label_key]
            else:
                logger.warning(f"unexpected result: {result}")
                label = ERROR_LABEL
            predicted_labels[_id] = label
        gold_labels, predicted_labels = (
            list(l)
            for l in zip(*[(gold_labels[_id], predicted_labels[_id]) for _id in ids])
        )

        return gold_label_eval(gold_labels, predicted_labels)
