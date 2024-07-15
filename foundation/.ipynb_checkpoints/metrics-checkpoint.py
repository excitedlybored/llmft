import datasets
import numpy as np
from transformers import EvalPrediction


metric_script = "../llmft/metrics/glue.py" 
# f"{os.environ['PROJECT_DIR']}/metrics/glue.py"

class CustomMetric:
    def __init__(self, task_name=None, dataset_cache_dir=None, metric_script=None, is_regression=False, target_tokens=None, target_tokens_logits_only=False, target_tokens_ids=None):
        self.task_name = task_name
        self.dataset_cache_dir = dataset_cache_dir
        self.metric_script = metric_script if metric_script else "/default/path/to/glue.py"
        self.metric = self.get_metrics()
        self.is_regression = is_regression
        self.target_tokens = target_tokens
        self.target_tokens_logits_only = target_tokens_logits_only
        self.target_tokens_ids = target_tokens_ids

    def get_metrics(self):
        if self.task_name is not None:
            if self.task_name == "mnli-original":
                metric = datasets.load_metric(path=self.metric_script, config_name="mnli",
                                              cache_dir=self.dataset_cache_dir, keep_in_memory=False)
            else:
                metric = datasets.load_metric(path=self.metric_script, config_name=self.task_name,
                                              cache_dir=self.dataset_cache_dir, keep_in_memory=False)
        else:
            metric = datasets.load_metric("accuracy", cache_dir=self.dataset_cache_dir, keep_in_memory=False)
        return metric

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.squeeze(
            preds) if self.is_regression else np.argmax(preds, axis=1)

        if self.task_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)

            # When using the lm_head, compute fraction of predictions that are not one of the target tokens
            if self.target_tokens is not None and not self.target_tokens_logits_only:
                unique_preds, counts_preds = np.unique(
                    preds, return_counts=True)
                unique_preds_counts_dict = dict(
                    zip(unique_preds, counts_preds))

                num_of_target_token_predictions = 0
                for idx in self.target_tokens_ids:
                    num_of_target_token_predictions += unique_preds_counts_dict.get(
                        idx, 0)
                num_other_tokens = len(
                    preds) - num_of_target_token_predictions
                result["frac_non_target_tokens"] = num_other_tokens / \
                    len(preds)

            # # Combine eval metrics
            # if len(result) > 1:
            #     result["combined_score"] = np.mean(
            #         list(result.values())).item()

            return result

        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
