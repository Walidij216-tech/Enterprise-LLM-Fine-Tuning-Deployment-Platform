
import os
os.environ["UNSLOTH_USE_FCEL"] = "0"

import json
from pathlib import Path
from dotenv import load_dotenv

from src.data.loader import DatasetLoader
from src.train.model_loader import ModelManager
from src.train.lora import LoRAConfig
from src.train.trainer import TrainerManager
from src.train.model_saver import ModelSaver
from src.experiments.mlflow_tracker import MLflowTracker


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

TRAINING_DATASET_PATH = os.getenv("TRAINING_DATASET_PATH")
VALIDATION_DATASET_PATH = os.getenv("VALIDATION_DATASET_PATH")


def main():

    print("\n" + "=" * 60)
    print("LLM FINE-TUNING PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Initialize managers
    # ---------------------------------------------------------

    print("\n[1/8] Initializing managers...")

    model_manager = ModelManager()
    trainer_manager = TrainerManager()

    train_dataset_loader = DatasetLoader(TRAINING_DATASET_PATH)
    val_dataset_loader = DatasetLoader(VALIDATION_DATASET_PATH)

    lora_config = LoRAConfig()

    model_saver = ModelSaver(
        gguf_dir=os.getenv("GGUF_MODEL_DIR")
    )

    mlflow_tracker = MLflowTracker()

    run_name = os.getenv(
        "MLFLOW_TRAINING_RUN_NAME",
        "Phi3_LoRA_Training"
    )

    # ---------------------------------------------------------
    # START ONE MLFLOW RUN FOR THE ENTIRE PIPELINE
    # ---------------------------------------------------------

    with mlflow_tracker.start_run(run_name):

        print("\nMLflow run started successfully.")

        # ---------------------------------------------------------
        # 2. Check GPU and display model information
        # ---------------------------------------------------------

        print("\n[2/8] Checking GPU and model configuration...")

        model_manager.check_gpu()

        model_info = model_manager.get_model_info()

        print("Model configuration:")
        print(model_info)

        # Log model information
        mlflow_tracker.set_tags({
            "stage": "training",
            "base_model": "Phi-3 Mini",
            "fine_tuning": "LoRA",
            "model_format": "GGUF"
        })

        if isinstance(model_info, dict):
            mlflow_tracker.log_params(model_info)

        # ---------------------------------------------------------
        # 3. Load datasets
        # ---------------------------------------------------------

        print("\n[3/8] Loading datasets...")

        train_data = train_dataset_loader.load_dataset()
        validation_data = val_dataset_loader.load_dataset()

        print(f"Training examples:   {len(train_data)}")
        print(f"Validation examples: {len(validation_data)}")

        # Log dataset information
        mlflow_tracker.log_params({
            "train_examples": len(train_data),
            "validation_examples": len(validation_data),
            "base_model": "Phi-3 Mini",
            "fine_tuning": "LoRA"
        })

        # ---------------------------------------------------------
        # 4. Format datasets
        # ---------------------------------------------------------

        print("\n[4/8] Formatting datasets...")

        train_dataset = model_manager.prepare_dataset(train_data)

        validation_dataset = model_manager.prepare_dataset(
            validation_data
        )

        print("Datasets formatted successfully.")

        # ---------------------------------------------------------
        # 5. Load base model
        # ---------------------------------------------------------

        print("\n[5/8] Loading base model...")

        model, tokenizer = model_manager.load_model()

        print("Base model loaded successfully.")

        # ---------------------------------------------------------
        # 6. Apply LoRA adapters
        # ---------------------------------------------------------

        print("\n[6/8] Applying LoRA adapters...")

        lora_config.display_config()
        lora_config.validate_config()

        model = lora_config.apply(model)

        print("LoRA adapters applied successfully.")

        # ---------------------------------------------------------
        # 7. Create trainer and start fine-tuning
        # ---------------------------------------------------------

        print("\n[7/8] Creating trainer...")

        trainer_manager.create_training_arguments()

        trainer = trainer_manager.create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset
        )

        # Display training configuration
        training_info = trainer_manager.get_training_info()

        print("\nTraining configuration:")
        print(json.dumps(training_info, indent=4, default=str))

        # Log training parameters
        mlflow_tracker.log_params(training_info)

        print("\n" + "=" * 60)
        print("STARTING FINE-TUNING")
        print("=" * 60)

        training_results = trainer_manager.train()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        print(training_results)

        # Log training metrics
        train_metrics = getattr(
            training_results,
            "metrics",
            training_results if isinstance(training_results, dict) else {}
        )

        mlflow_tracker.log_metrics(
            train_metrics,
            prefix="train."
        )

        # ---------------------------------------------------------
        # 8. Evaluate and save fine-tuned model
        # ---------------------------------------------------------

        print("\n[8/8] Evaluating and saving the model...")

        # Validation evaluation
        validation_results = trainer_manager.evaluate()

        print("\nValidation results:")
        print(validation_results)

        # Log validation metrics
        mlflow_tracker.log_metrics(
            validation_results,
            prefix="validation."
        )

        # Save validation results as an MLflow artifact
        if isinstance(validation_results, dict):
            mlflow_tracker.log_dict(
                validation_results,
                "validation_results.json"
            )

        # Save training configuration as an MLflow artifact
        mlflow_tracker.log_dict(
            training_info,
            "training_configuration.json"
        )

        # Save the fine-tuned model in GGUF format
        saved_paths = model_saver.save_model(
            trainer.model,
            tokenizer
        )

        print("\nSaved model locations:")
        print(json.dumps(saved_paths, indent=4))

        # Record export information
        mlflow_tracker.log_params({
            "model_format": "GGUF",
            "quantization": "Q4_K_M"
        })

        # Record GGUF export path as a tag
        mlflow_tracker.set_tags({
            "gguf_export": json.dumps(saved_paths)
        })

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

    # ---------------------------------------------------------
    # MLflow run is automatically closed here
    # ---------------------------------------------------------

    print("\nMLflow run completed and closed successfully.")


if __name__ == "__main__":
    main()

