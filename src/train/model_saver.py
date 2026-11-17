

from pathlib import Path


class ModelSaver:
    """
    Save and export the fine-tuned LLM.

    Supported format:
    - GGUF (Q4_K_M)
    """

    def __init__(self, gguf_dir="gguf_model"):
        self.gguf_dir = Path(gguf_dir)

    def save_model(self, model, tokenizer):
        """
        Export the fine-tuned model into GGUF format.

        Args:
            model: Fine-tuned model.
            tokenizer: Model tokenizer.

        Returns:
            Dictionary containing the exported model path.
        """

        if model is None or tokenizer is None:
            raise ValueError(
                "Model and tokenizer must not be None."
            )

        # Create output directory
        self.gguf_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Export model to GGUF
        print("\n" + "=" * 60)
        print("EXPORTING FINE-TUNED MODEL TO GGUF")
        print("=" * 60)

        model.save_pretrained_gguf(
            str(self.gguf_dir),
            tokenizer,
            quantization_method="q4_k_m"
        )

        # Unsloth generates the GGUF file in a sibling directory
        gguf_output_dir = Path(
            f"{self.gguf_dir}_gguf"
        )

        # Verify GGUF export
        gguf_files = list(
            gguf_output_dir.rglob("*.gguf")
        )

        if not gguf_files:
            raise FileNotFoundError(
                f"No GGUF file found in "
                f"{gguf_output_dir.resolve()}."
            )

        # Select the Q4_K_M quantized model
        quantized_files = [
            file for file in gguf_files
            if "Q4_K_M" in file.name.upper()
        ]

        if not quantized_files:
            raise FileNotFoundError(
                f"No Q4_K_M GGUF file found in "
                f"{gguf_output_dir.resolve()}."
            )

        gguf_file = quantized_files[0]

        print("\nGGUF export completed successfully!")
        print(f"GGUF model: {gguf_file.resolve()}")

        return {
            "gguf_model": str(gguf_file.resolve())
        }
