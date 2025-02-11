from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
from textSummarizer.entity import ModelTrainerConfig
from transformers import AdamW




class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):

        tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model)

        for param in model_pegasus.parameters():
            param.requires_grad = False

        for param in model_pegasus.model.encoder.layers[-2:].parameters():
            param.requires_grad = True

        for param in model_pegasus.model.decoder.layers[-2:].parameters():
            param.requires_grad = True

        dataset = load_from_disk(self.config.data_path)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps, learning_rate=5e-5
            ) 

        optimizer = AdamW([
            {"params": model_pegasus.model.encoder.layers[-2:].parameters(), "lr": 3e-5},
            {"params": model_pegasus.model.decoder.layers[-2:].parameters(), "lr": 3e-5},
        ], lr=5e-5, weight_decay=self.config.weight_decay)

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            optimizers=(optimizer, None)
        )

        trainer.train()

        # Save the fine-tuned model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))