from src.config.evaluation_config import EvaluationConfig



class MetricsCalculator:
    """
    Calculate evaluation metrics for structured
    product information extraction.
    """

    REQUIRED_FIELDS = EvaluationConfig.REQUIRED_FIELDS


    @staticmethod
    def normalize(value):
        """
        Normalize values before comparison.
        """
        if value is None:
            return ""

        return str(value).strip().lower()

    @classmethod
    def calculate_metrics(cls, predictions, references):#cls refers to the class itself with the @classmethod decorator.

        if len(predictions) != len(references):
            raise ValueError(
                "Predictions and references must have the same length."
            )

        if not references:
            raise ValueError("Evaluation dataset is empty.")

        total_fields = 0
        correct_fields = 0
        exact_matches = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0

        valid_json = 0

        for prediction, reference in zip(
            predictions, references
        ):

            if prediction.get("json_valid", False): #If the key exists, return its value.If the key does not exist, return False
                valid_json += 1  

            predicted_output = prediction.get("output")

            if not isinstance(predicted_output, dict):
                predicted_output = {}

            example_correct = True

            for field in cls.REQUIRED_FIELDS:

                predicted_value = cls.normalize(
                    predicted_output.get(field)
                )

                expected_value = cls.normalize(
                    reference.get(field)
                )

                total_fields += 1

                if predicted_value == expected_value:
                    correct_fields += 1

                    if expected_value:#if expected_value exists
                        true_positive += 1

                else:
                    example_correct = False

                    if predicted_value:
                        false_positive += 1
                        #FP:The model produced a value, but it is incorrect.
                    if expected_value:
                        false_negative += 1
                        #FN:The expected price exists, but the model failed to extract it.
            if example_correct:
                exact_matches += 1

        accuracy = correct_fields / total_fields

        exact_match_accuracy = (
            exact_matches / len(references)
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        json_validity = valid_json / len(references)

        return {
            "field_accuracy": accuracy,
            "exact_match_accuracy": exact_match_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "json_validity": json_validity,
            "total_examples": len(references),
            "correct_examples": exact_matches,
            "valid_json_examples": valid_json
        }