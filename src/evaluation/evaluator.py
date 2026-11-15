import json
import requests
from src.config.evaluation_config import EvaluationConfig


class ModelEvaluator:
    """
    Evaluate a fine-tuned model deployed through Ollama.
    """

    def __init__(self):

        self.model_name = EvaluationConfig.MODEL_NAME

        self.api_url = (
            f"{EvaluationConfig.API_URL}/api/generate"
        )

        self.temperature = EvaluationConfig.TEMPERATURE

    @staticmethod
    def build_prompt(html_input):
        """
        Build an inference prompt consistent with training.
        """

        return (
            f"### Input: {html_input}\n"
            "### Output:"
        )

    @staticmethod
    def parse_response(response_text):
        """
        Parse the model response into a Python dictionary.
        """

        try:
            parsed = json.loads(response_text)

            if isinstance(parsed, dict):
                return parsed, True

            return {}, False

        except (json.JSONDecodeError, TypeError):
            return {}, False

    def predict(self, html_input):
        """
        Generate a prediction using Ollama.
        """

        prompt = self.build_prompt(html_input)

        payload = {
              "model": self.model_name,
              "prompt": prompt,
              "stream": False,
              "format": "json",
              "options": {
              "temperature": self.temperature
             }
                  }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=180
            )

            response.raise_for_status()

            result = response.json()

            raw_output = result.get("response", "")

            parsed_output, json_valid = (
                self.parse_response(raw_output)
            )

            return {
                "output": parsed_output,
                "raw_output": raw_output,
                "json_valid": json_valid,
                "error": None
            }
            '''
            prediction = {
    "output": {
        "name": "Samsung Galaxy S24",
        "price": "$799",
        "category": "Smartphone",
        "manufacturer": "Samsung"
    },
    "raw_output": "{\"name\": \"Samsung Galaxy S24\"}",
    "json_valid": True,
    "error": None
             }                            '''

        except requests.RequestException as error:

            return {
                "output": {},
                "raw_output": "",
                "json_valid": False,
                "error": str(error)
            }

    def evaluate_dataset(self, test_dataset):
        """
        Generate predictions for all test examples.
        """

        predictions = []

        for index, example in enumerate(test_dataset):

            print(
                f"Evaluating example "
                f"{index + 1}/{len(test_dataset)}"
            )

            prediction = self.predict(
                example["input"]
            )
            print(prediction)
            prediction["example_id"] = index + 1 #creates a new key in the prediction dictionary

            predictions.append(prediction)

        return predictions