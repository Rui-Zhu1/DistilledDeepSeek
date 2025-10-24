# DistilledDeepSeek Fine-Tuning

This project provides a script to fine-tune the `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model on a niche dataset about vintage camera restoration.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rui-Zhu1/DistilledDeepSeek.git
    cd DistilledDeepSeek
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Log in to Weights & Biases:**
    ```bash
    wandb login
    ```

## Running the Script

To run the fine-tuning script, execute the following command:

```bash
python run_finetuning.py
```

The script will:
1.  Create the niche dataset.
2.  Load the pre-trained model and tokenizer.
3.  Tokenize the dataset.
4.  Fine-tune the model using LoRA and 8-bit quantization.
5.  Save the fine-tuned model to the `deepseek_finetuned_full` directory.
