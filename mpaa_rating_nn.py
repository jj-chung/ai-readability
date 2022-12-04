import numpy as np
import matplotlib.pyplot as plt
import imbalanced
import data_preprocessing
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate

if __name__ == "__main__":
    # Getting data 
    dataset_train = data_preprocessing.bert_pre_processing(type="train")
    dataset_test = data_preprocessing.bert_pre_processing(type="test")

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()