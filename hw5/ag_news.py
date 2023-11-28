import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def load_text(file_path):
    data = pd.read_csv(file_path)
    data.columns = ['Class', 'Title', 'Description']
    data = data.sample(frac=1.).reset_index(drop=True)

    ds = data.copy()
    ds['text'] = data['Title'] + ' ' + data['Description']
    ds.rename(columns={'Class': 'label'}, inplace=True)
    ds['label'] = ds['label'] - 1
    ds.drop(['Title', 'Description'], axis=1, inplace=True)

    return ds


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def pipeline(dataframe):
    dataset = Dataset.from_pandas(dataframe, preserve_index=False)
    tokenized_ds = dataset.map(preprocess_function, batched=True)
    tokenized_ds = tokenized_ds.remove_columns('text')
    return tokenized_ds


# requires pytorch by default // tried to make it work for tensorflow but the old way deprecated and
# requires keras
# https://www.kaggle.com/code/keithcooper/multi-class-classification-with-transformer-models
if __name__ == '__main__':
    print('GPU availability: ', torch.cuda.is_available())

    ds_train = load_text('./train.csv')
    ds_test = load_text('./test.csv')

    ds_train, ds_val = train_test_split(ds_train[['text', 'label']], test_size=0.2, random_state=42, shuffle=True)

    # max length 177 and 68 is +3 std of the length
    pretrained_model = "microsoft/MiniLM-L12-H384-uncased"  # best training efficiency for one epoch
    # pretrained_model = 'google/electra-base-generator'  # 'lightweight' but converges more slowly

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    tokenized_train = pipeline(ds_train)
    tokenized_val = pipeline(ds_val)
    tokenized_test = pipeline(ds_test)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=4)

    training_args = TrainingArguments(
        output_dir="./results",
        save_strategy='epoch',
        optim="adamw_torch",
        learning_rate=2e-5,
        # somehow bigger batch size trains more slowly for one epoch
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        report_to="none",  # Stops transformers from trying to connect to weights and biases site
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    y_test = tokenized_test['label']
    tokenized_test = tokenized_test.remove_columns('label')

    preds = trainer.predict(tokenized_test)
    preds_flat = [np.argmax(x) for x in preds[0]]

    bool_count = np.isclose(preds_flat, y_test)
    correct = bool_count.sum()
    accuracy = correct/len(preds_flat)

    print('Test set accuracy: ', accuracy)  # test accuracy 0.9330
