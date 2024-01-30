import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Define the dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Define the model
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256 * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        lstm_output, _ = self.lstm(roberta_output[0])
        pooled_output = torch.mean(lstm_output, 1)
        return self.classifier(pooled_output)


# Load the data
df = pd.read_csv('train.csv').iloc[0:2000,]
X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.label.values, test_size=0.1, random_state=17, stratify=df.label.values)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_len = 128 # or choose a maximum length based on your dataset

# Create the data loaders
train_dataset = SentimentDataset(
    texts=df.sentence.to_numpy(),
    labels=df.label.to_numpy(),  # Use 'label' instead of 'sentiment'
    tokenizer=tokenizer,
    max_len=max_len
)

# Rest of the code remains the same...

train_data_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentClassifier(n_classes=2)
model = model.to(device)

# Training hyperparameters
EPOCHS = 300
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss().to(device)

from sklearn.metrics import classification_report

# Training loop with classification report
for epoch in range(EPOCHS):
    # Lists to hold actual and predicted labels
    true_labels = []
    pred_labels = []

    for data in train_data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)  # Labels are already tensors

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        # Move preds to CPU for sklearn metrics
        preds = preds.detach().cpu().numpy()
        # Labels are already tensors, so we just get them to CPU
        labels = labels.detach().cpu().numpy()

        # Append batch prediction results
        pred_labels.extend(preds)
        true_labels.extend(labels)

        # Convert labels back to tensor for loss calculation
        labels = torch.tensor(labels).to(device)  # Convert labels back to tensor

        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Calculate classification report
    class_report = classification_report(true_labels, pred_labels, target_names=['Negative', 'Positive'], digits=4)
    print(f'Epoch {epoch + 1}/{EPOCHS} finished')
    print(class_report)


