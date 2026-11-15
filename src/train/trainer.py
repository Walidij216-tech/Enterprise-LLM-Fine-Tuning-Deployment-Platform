# src/training/trainer.py


import os
import torch

from dotenv import load_dotenv
from trl import SFTTrainer
from transformers import TrainingArguments


class TrainerManager:
    """
    Manage SFT training configuration, trainer creation
    and fine-tuning execution.
    """

    def __init__(self, env_path=".env"):

        # Load environment variables
        load_dotenv(dotenv_path=env_path)

        # Training configuration
        self.output_dir = os.getenv("OUTPUT_DIR", "outputs")

        self.per_device_train_batch_size = int(
            os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE")
        )

        self.gradient_accumulation_steps = int(
            os.getenv("GRADIENT_ACCUMULATION_STEPS")
        )

        self.num_train_epochs = int(
            os.getenv("NUM_TRAIN_EPOCHS")
        )

        self.learning_rate = float(
            os.getenv("LEARNING_RATE")
        )

        self.warmup_steps = int(
            os.getenv("WARMUP_STEPS")
        )

        self.logging_steps = int(
            os.getenv("LOGGING_STEPS")
        )

        self.weight_decay = float(
            os.getenv("WEIGHT_DECAY")
        )

        self.seed = int(
            os.getenv("SEED")
        )

        self.dataset_num_proc = int(
            os.getenv("DATASET_NUM_PROC")
        )

        self.max_seq_length = int(
            os.getenv("MAX_SEQ_LENGTH")
        )

        self.trainer = None

    # --------------------------------------------------
    # 1. GPU PRECISION CONFIGURATION
    # --------------------------------------------------

    @staticmethod
    def get_precision_config():
        """
        Detect GPU precision support and configure FP16/BF16.
        """

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for this training configuration."
            )

        bf16_supported = torch.cuda.is_bf16_supported()

        return {
            "fp16": not bf16_supported,
            "bf16": bf16_supported
        }

    # --------------------------------------------------
    # 2. TRAINING ARGUMENTS
    # --------------------------------------------------

    def create_training_arguments(self):
        """
        Create Hugging Face TrainingArguments.
        """

        precision_config = self.get_precision_config()

        return TrainingArguments(

            output_dir=self.output_dir,

            per_device_train_batch_size=(
                self.per_device_train_batch_size
            ),

            per_device_eval_batch_size=(
                self.per_device_train_batch_size
            ),

            gradient_accumulation_steps=(
                self.gradient_accumulation_steps
            ),

            warmup_steps=self.warmup_steps,

            num_train_epochs=self.num_train_epochs,

            learning_rate=self.learning_rate,

            fp16=precision_config["fp16"],

            bf16=precision_config["bf16"],

            logging_steps=self.logging_steps,

            optim="adamw_8bit",

            weight_decay=self.weight_decay,

            lr_scheduler_type="linear",

            seed=self.seed,

            save_strategy="epoch",

            save_total_limit=2,

            dataloader_pin_memory=False,

            report_to="none",

            eval_strategy="epoch",
        )

    # --------------------------------------------------
    # 3. CREATE SFT TRAINER
    # --------------------------------------------------

    def create_trainer(
        self,
        model,
        tokenizer,
        train_dataset,
        validation_dataset
    ):
        """
        Create and configure the TRL SFTTrainer.
        """

        if model is None or tokenizer is None:
            raise ValueError(
                "Model and tokenizer must not be None."
            )

        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        if len(validation_dataset) == 0:
            raise ValueError("Validation dataset is empty.")

        print("\n" + "=" * 60)
        print("CREATING SFT TRAINER")
        print("=" * 60)

        print(f"Training samples:   {len(train_dataset)}")
        print(f"Validation samples: {len(validation_dataset)}")

        effective_batch_size = (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
        )

        print(f"Effective batch size: {effective_batch_size}")

        training_args = self.create_training_arguments()

        self.trainer = SFTTrainer(
            model=model,

            tokenizer=tokenizer,

            train_dataset=train_dataset,

            eval_dataset=validation_dataset,

            dataset_text_field="text",

            max_seq_length=self.max_seq_length,

            dataset_num_proc=self.dataset_num_proc,

            args=training_args,
        )

        print("\nSFTTrainer created successfully.")

        return self.trainer

    # --------------------------------------------------
    # 4. START TRAINING
    # --------------------------------------------------

    def train(self):
        """
        Start the fine-tuning process.
        """

        if self.trainer is None:
            raise RuntimeError(
                "Trainer not initialized. "
                "Call create_trainer() first."
            )

        print("\n" + "=" * 60)
        print("STARTING FINE-TUNING")
        print("=" * 60)

        training_results = self.trainer.train()

        print("\nFine-tuning completed successfully.")

        return training_results

    # --------------------------------------------------
    # 5. EVALUATE MODEL
    # --------------------------------------------------

    def evaluate(self, test_dataset=None):
        """
        Evaluate the trained model.

        If test_dataset is provided, evaluate on that dataset.
        Otherwise, evaluate on the validation dataset.
        """

        if self.trainer is None:
            raise RuntimeError(
                "Trainer not initialized."
            )

        if test_dataset is not None:
            results = self.trainer.evaluate(
                eval_dataset=test_dataset
            )
        else:
            results = self.trainer.evaluate()

        print("\nEvaluation results:")
        print(results)

        return results

    # --------------------------------------------------
    # 6. GET TRAINING CONFIGURATION
    # --------------------------------------------------

    def get_training_info(self):
        """
        Return training hyperparameters.
        """

        return {
            "output_dir": self.output_dir,
            "train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": (
                self.per_device_train_batch_size
                * self.gradient_accumulation_steps
            ),
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "max_seq_length": self.max_seq_length,
        }
