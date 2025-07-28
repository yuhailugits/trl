"""GRPO config for next-edit prediction training using TRL."""
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

def train_next_edit(model_path: str, dataset, output_dir: str):
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        kl_coef=0.1,
        max_new_tokens=128,
        temperature=0.8,
        logging_steps=10,
    )
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    trainer = GRPOTrainer(model=model, args=config, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(output_dir)

