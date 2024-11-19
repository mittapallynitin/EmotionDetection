import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

device = "mps"
checkpoint = "prajjwal1/bert-tiny"

df = pd.read_csv("./data/text.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)
df.head()

label_to_id = {
    "sadness":0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5
}
n_classes = len(label_to_id)
id_to_label = {v:k for k,v in label_to_id.items()}
df["emotion"] = df["label"].map(id_to_label)



tokenizer = AutoTokenizer.from_pretrained(checkpoint)
bert_model = AutoModel.from_pretrained(checkpoint).to(device)



class EmotionDataset(Dataset):

    def __init__(self, df, tokenizer, max_len=512):
        self.text = df.text.tolist()
        self.labels = df.label.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        return {
            "text": text,
            "labels": self.labels[index]
        }


class EmotionCollator:
    def __init__(self, tokenizer, max_len=512, device="mps"):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def collate_fn(self, batch):
        sentence_batch = [example["text"] for example in batch]
        labels_batch = [example["labels"] for example in batch]

        inputs = self.tokenizer(
            sentence_batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_len
        )

        labels = torch.tensor(labels_batch, dtype=torch.long)

        return {
            'input_ids': inputs['input_ids'].to(device),
            'attention_mask': inputs['attention_mask'].to(device),
            'labels': labels.to(device)
        }


train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=43,
    stratify=df.emotion,
    shuffle=True
)

train_ds = EmotionDataset(train_df, tokenizer)
test_ds = EmotionDataset(test_df, tokenizer)
emotion_collator = EmotionCollator(tokenizer)

train_dataloader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    collate_fn=emotion_collator.collate_fn
)
test_dataloader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=True,
    collate_fn=emotion_collator.collate_fn
)


class EmotionClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, expansion_factor=2):
        super(EmotionClassifier, self).__init__()
        self.bert_model = bert_model
        H = bert_model.config.hidden_size
        self.linear1 = nn.Linear(H, H * expansion_factor)
        self.linear2 = nn.Linear(H * expansion_factor, num_labels)

    def forward(self, inputs):
        x = self.bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).pooler_output
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

classifier_model = EmotionClassifier(bert_model, num_labels=n_classes).to(device)
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
epochs = 5



def train_model(model, optimizer, criterion, batch):
    optimizer.zero_grad()
    outputs = model(batch)
    loss = criterion(outputs, batch["labels"])
    loss.backward()
    optimizer.step()
    return loss

def evaluate_model(model, criterion, batch):
    with torch.no_grad():
        outputs = model(batch)
        loss = criterion(outputs, batch["labels"])
    return loss


train_loss = []
val_loss = []


for epoch in range(epochs):
    total_loss = 0
    for train_batch_idx, batch in enumerate(train_dataloader):
        classifier_model.train()
        loss = train_model(classifier_model, optimizer, criterion, batch)
        total_loss += loss.sum().item()

        if train_batch_idx % 10 == 0:
            train_loss.append(total_loss)
            classifier_model.eval()
            total_loss = 0
            for val_batch_idx, batch in enumerate(test_dataloader):
                loss = evaluate_model(classifier_model, criterion, batch)
                total_loss += loss.sum().item()
                if val_batch_idx % 10 == 0:
                    break
            val_loss.append(total_loss)
            total_loss = 0
            print("train_loss", train_loss[-1], "val_loss", val_loss[-1])

# Save the trained model
torch.save(classifier_model.state_dict(), "classifier_model.pt")
