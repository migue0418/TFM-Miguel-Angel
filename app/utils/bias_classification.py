from app.schemas.redditbias import Topic
from app.schemas.bias_classification import BiasDataset
from app.core.config import redditbias_data_path as data_path
from app.core.config import files_path
from app.core.config import settings
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from tqdm import tqdm


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())

    return total_loss / len(data_loader), all_labels, all_preds


def train_bias_classificator(topic: Topic = None, edos: bool = False):
    # Load the dataset
    if edos:
        folder_path = "edos"
        processed_file_path = files_path / "edos_labelled_aggregated.csv"
    else:
        folder_path = topic.name
        processed_file_path = (
            data_path
            / topic.name
            / f"reddit_comments_{topic.name}_{topic.minority_group}_processed_phrase_annotated.csv"
        )
    data = pd.read_csv(processed_file_path)

    if edos:
        data = data[["text", "label_sexist"]].rename(
            columns={"text": "comment", "label_sexist": "bias_sent"}
        )
        # Convertir los valores de bias_sent: 'sexist' -> 1, 'not sexist' -> 0
        data["bias_sent"] = data["bias_sent"].map({"sexist": 1, "not sexist": 0})

    # Use the columns `comment` as text and `bias_sent` as label
    comments = data["comment"].values
    labels = data["bias_sent"].values

    # Create a train / test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        comments, labels, test_size=0.2, random_state=42
    )

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = BiasDataset(train_texts, train_labels, tokenizer)
    val_dataset = BiasDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Get the model with a binary output
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_path = settings.BIAS_MODEL_PATH + "/" + folder_path

    # Training
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = BCEWithLogitsLoss()

    # Train for 3 epochs
    epochs = 3
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_labels, val_preds = eval_model(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the model
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def get_pretrained_model_bias_classificator(
    topic_ins: Topic = None, force_training: bool = False, edos: bool = False
):
    """Get the pretrained model bias classificator and tokenizer or train a new one"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_path = topic_ins.name if not edos else "edos"
    model_path = settings.BIAS_MODEL_PATH + "/" + folder_path

    if os.path.exists(model_path) and not force_training:
        print("Modelo encontrado. Cargando...")
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    else:
        print("Entrenando un nuevo modelo...")
        train_bias_classificator(topic_ins, edos)
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()
    return device, model, tokenizer


def predict_bias(
    topic_ins: Topic, text: str, force_training: bool = False, edos: bool = False
):
    """Realiza una predicción de sesgo dado un texto."""
    device, model, tokenizer = get_pretrained_model_bias_classificator(
        topic_ins, force_training, edos=edos
    )
    encoding = tokenizer(
        text, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        prob = torch.sigmoid(logits).item()

    return prob


def read_phrases_files():
    """Read the phrases files with the polarities"""
    files = {
        "high_bias": "high_bias.txt",
        "neutral": "neutral.txt",
        "small_bias": "small_bias.txt",
    }
    scores = {"high_bias": 1, "neutral": 0, "small_bias": 0.5}

    data = []
    for bias_type, filename in files.items():
        file_path = os.path.join(files_path, "phrases_examples", filename)
        with open(file_path, "r", encoding="utf-8") as file:
            phrases = file.readlines()
            for phrase in phrases:
                data.append({"text": phrase.strip(), "manual_score": scores[bias_type]})

    return data
