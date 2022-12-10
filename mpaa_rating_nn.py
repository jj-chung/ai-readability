import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
import torch

#NOTE: DO NOT RUN THIS CODE. It is out of date. Instead run mpaa_bert_nn.ipynb on GPU

if __name__ == "__main__":
    
    #print(torch.cuda.is_available())
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Getting data 
    # By default, the Trainer will use the GPU if it is available. https://discuss.huggingface.co/t/sending-a-dataset-or-datasetdict-to-a-gpu/17208
    dataset_train = data_preprocessing.bert_pre_processing(type="train", sample_type = "RUS")
    dataset_test = data_preprocessing.bert_pre_processing(type="test")

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    # model.to(device)
    # print(model.device)

    training_args = TrainingArguments(
        output_dir="test_trainer", 
        evaluation_strategy="epoch",
        num_train_epochs = 3,
        gradient_accumulation_steps = 1,
        per_device_train_batch_size = 8,
        learning_rate = 5e-5
    )
    metric = evaluate.combine([
        evaluate.load("accuracy", average="micro"), 
        evaluate.load("recall", average="micro"), 
        evaluate.load("precision", average="micro"), 
        evaluate.load("f1", average="micro")
        ])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="micro")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()