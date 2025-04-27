from app.enums.datasets_enums import DatasetEnum
from app.enums.models_enums import ModelsEnum
from app.schemas.redditbias import Topic
from app.schemas.bias_classification import BiasDataset
from app.core.config import redditbias_data_path as data_path
from app.core.config import files_path
from app.core.config import settings
import pandas as pd
import os
import torch
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
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
    # Cargamos el dataset
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
        # Convertir los valores de bias_sent a 1/0
        data["bias_sent"] = data["bias_sent"].map({"sexist": 1, "not sexist": 0})

    # Obtenemos los comentarios y etiquetas
    comments = data["comment"].values
    labels = data["bias_sent"].values

    # Partimos en train/val (80/20)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        comments, labels, test_size=0.2, random_state=42
    )

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = BiasDataset(train_texts, train_labels, tokenizer)
    val_dataset = BiasDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Cargar el modelo
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_path = settings.BIAS_MODEL_PATH + "/" + folder_path

    # Entrenar el modelo
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = BCEWithLogitsLoss()

    # Usamos 3 epochs
    epochs = 3
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_labels, val_preds = eval_model(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Guardamos el modelo
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def train_model(dataset: DatasetEnum, model_name: ModelsEnum):
    """
    Entrena un modelo para detección de sexismo (2 clases: sexist / not sexist).

    - Si es un dataset de EDOS, se asume un CSV con columnas:
      ['text', 'label_sexist', 'split', ...] y usa train/dev/test de la columna 'split'.
    - Si es de RedditBias, se asume un CSV con columnas
      ['comment', 'bias_sent', ...] y se hace un train_test_split del 80/20.
    """
    # Obtener ruta donde se guardará el modelo
    model_path = os.path.join(
        settings.BIAS_MODEL_PATH,
        dataset.model_folder_path,
        model_name.value.split("/")[-1],
    )

    # Cargar el dataset desde CSV
    df = pd.read_csv(dataset.csv_path)

    # Procesar de forma distinta según sea EDOS o no
    if "edos" in dataset.name:
        # Asegurar que las columnas 'text', 'label_sexist' y 'split' existen
        if not all(col in df.columns for col in ["text", "label_sexist", "split"]):
            raise ValueError(
                "El CSV de EDOS debe contener las columnas 'text', 'label_sexist' y 'split'."
            )

        # Crear splits a partir de la columna 'split'
        df_train = df[df["split"] == "train"]
        df_dev = df[df["split"] == "dev"]
        df_test = df[df["split"] == "test"]

        # Renombrar 'text' -> 'comment' y 'label_sexist' -> 'label' (por convención en HF)
        df_train = df_train.rename(columns={"text": "comment", "label_sexist": "label"})
        df_dev = df_dev.rename(columns={"text": "comment", "label_sexist": "label"})
        df_test = df_test.rename(columns={"text": "comment", "label_sexist": "label"})

        # Convertir cada split en Dataset
        dataset_train = Dataset.from_pandas(df_train)
        dataset_dev = Dataset.from_pandas(df_dev)
        dataset_test = Dataset.from_pandas(df_test)

    else:
        # Convertir bias_sent a entero si no lo es
        df["bias_sent"] = df["bias_sent"].astype(int)

        # Crear dataset a partir del dataframe
        df = df[["comment", "bias_sent"]].dropna()
        df = df.rename(columns={"bias_sent": "labels"})
        dataset_full = Dataset.from_pandas(df)

        # Dividir en entrenamiento (80%) y validación/test (20%)
        dataset_full = dataset_full.train_test_split(test_size=0.2, seed=42)
        dataset_train = dataset_full["train"]
        dataset_dev = dataset_full["test"]

        # No hay test separado en este caso.
        dataset_test = None

    # Convertir la columna "label" a un tipo ClassLabel (string -> int)
    dataset_train = dataset_train.class_encode_column("label")
    dataset_dev = dataset_dev.class_encode_column("label")
    if dataset_test is not None:
        dataset_test = dataset_test.class_encode_column("label")

    # Cargar el tokenizador y modelo base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Configurar 'label2id' e 'id2label' en el modelo para que devuelva
    #  directamente las etiquetas de texto
    label_names = dataset_train.features["label"].names
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for i, name in enumerate(label_names)}
    model.config.label2id = label2id
    model.config.id2label = id2label

    # Función de tokenización
    def tokenize_function(examples):
        return tokenizer(
            examples["comment"], padding="max_length", truncation=True, max_length=128
        )

    # Tokenizamos cada subset
    dataset_train = dataset_train.map(tokenize_function, batched=True)
    dataset_dev = dataset_dev.map(tokenize_function, batched=True)
    if dataset_test is not None:
        dataset_test = dataset_test.map(tokenize_function, batched=True)

    # Eliminamos columnas que sobran
    dataset_train = dataset_train.remove_columns(["comment"])
    dataset_dev = dataset_dev.remove_columns(["comment"])
    if dataset_test is not None:
        dataset_test = dataset_test.remove_columns(["comment"])

    # Cargamos las métricas
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "precision": precision_metric.compute(predictions=preds, references=labels)[
                "precision"
            ],
            "recall": recall_metric.compute(predictions=preds, references=labels)[
                "recall"
            ],
            "f1": f1_metric.compute(predictions=preds, references=labels)["f1"],
        }

    # Configurar argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="epoch",  # evalúa al final de cada época
        save_strategy="epoch",  # guarda el modelo al final de cada época si mejora
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=11,
        weight_decay=0.001,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=100,
        seed=42,
        fp16=True,  # uso de GPU, acelera el entrenamiento
    )

    # Crear y ejecutar el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Entrenar el modelo
    trainer.train()

    # Evaluar en test (si existe dataset_test)
    if dataset_test is not None and len(dataset_test) > 0:
        print("Evaluando en el conjunto de test...")
        test_metrics = trainer.evaluate(dataset_test)
        print("Test metrics:", test_metrics)

    # Guardar modelo y tokenizador
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Modelo y tokenizador guardados en {model_path}")
    print(f"label2id: {model.config.label2id}")
    print(f"id2label: {model.config.id2label}")


def get_pretrained_model_bias_classificator(
    topic_ins: Topic = None, force_training: bool = False, edos: bool = False
):
    """Get the pretrained model bias classificator and tokenizer or train a new one"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_path = topic_ins.name if not edos else "edos"
    model_path = settings.BIAS_MODEL_PATH + "/" + folder_path

    if os.path.exists(model_path) and not force_training:
        print("Modelo encontrado. Cargando...")
    else:
        # Si no existe el modelo o si forzamos reentrenamiento
        print("Entrenando un nuevo modelo...")
        train_bias_classificator(topic_ins, edos)

    # Cargamos el modelo
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()
    return device, model, tokenizer


def predict_bias(
    topic_ins: Topic, text: str, force_training: bool = False, edos: bool = False
):
    """Realiza una predicción de sesgo dado un texto."""
    # Cargar el modelo y el tokenizador
    device, model, tokenizer = get_pretrained_model_bias_classificator(
        topic_ins, force_training, edos=edos
    )
    encoding = tokenizer(
        text, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        # Realizar la predicción
        logits = model(input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        prob = torch.sigmoid(logits).item()

    return prob


def read_phrases_files():
    """Read the phrases files with the polarities and redurn a list of dictionaries"""
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


def reduce_edos_dataset():
    # Load the dataset
    processed_file_path = files_path / "edos_labelled_aggregated.csv"
    output_path = files_path / "edos_labelled_reduced_10k.csv"

    # Read it with pandas
    data = pd.read_csv(processed_file_path)

    # Get only 5k samples of the dataset with balanced sexists and non-sexists (2.5k for each)
    data_sexist = data[data["label_sexist"] == "sexist"].sample(n=4854, random_state=42)
    data_not_sexist = data[data["label_sexist"] == "not sexist"].sample(
        n=4854, random_state=42
    )

    # Concatenate the two dataframes
    data_reduced = pd.concat([data_sexist, data_not_sexist])

    # Save the reduced dataset
    data_reduced.to_csv(output_path, index=False)

    return data_reduced


def generate_consensus_datasets():
    # Load the dataset
    csv_path = files_path / "edos_labelled_individual_annotations.csv"

    # Read it with pandas
    df = pd.read_csv(csv_path)

    # Filter the test samples
    df_test = df[df["split"] == "test"]

    # Agrupar por rewire_id y recolectar las etiquetas en una lista
    grouped = (
        df_test.groupby("rewire_id")["label_sexist"]
        .apply(list)
        .reset_index(name="annot_labels")
    )

    # Clasificar cada rewire_id en base a las 3 anotaciones
    def classify_labels(label_list):
        unique = set(label_list)
        if unique == {"sexist"}:
            return "all_sexist"
        elif unique == {"not sexist"}:
            return "all_not_sexist"
        else:
            return "mixed"

    # Aplicar la función de clasificación a cada fila
    grouped["consensus"] = grouped["annot_labels"].apply(classify_labels)

    # Carga el dataset de aggregated y obtiene la etiqueta de ahí uniéndolo con grouped
    df_aggregated = pd.read_csv(files_path / "edos_labelled_aggregated.csv")

    df_test_merged = pd.merge(
        df_aggregated, grouped[["rewire_id", "consensus"]], on="rewire_id", how="left"
    )

    # Finalmente separamos en 3 dataframes (por unanimidad vs. mixto):
    df_test_all_sexist = df_test_merged[
        df_test_merged["consensus"] == "all_sexist"
    ].copy()
    df_test_all_not_sexist = df_test_merged[
        df_test_merged["consensus"] == "all_not_sexist"
    ].copy()
    df_test_mixed = df_test_merged[df_test_merged["consensus"] == "mixed"].copy()

    # Agrupar por rewire_id para evitar repetidos
    df_test_all_sexist = df_test_all_sexist.groupby("rewire_id").first().reset_index()
    df_test_all_not_sexist = (
        df_test_all_not_sexist.groupby("rewire_id").first().reset_index()
    )
    df_test_mixed = df_test_mixed.groupby("rewire_id").first().reset_index()

    # Vemos los IDs únicos de cada dataframe
    print("All sexist:", df_test_all_sexist["rewire_id"].nunique())
    print("All not sexist:", df_test_all_not_sexist["rewire_id"].nunique())
    print("Mixed:", df_test_mixed["rewire_id"].nunique())

    # Guarda los datasets como CSVs separados
    df_test_all_sexist.to_csv(files_path / "edos_labelled_all_sexist.csv", index=False)
    df_test_all_not_sexist.to_csv(
        files_path / "edos_labelled_all_not_sexist.csv", index=False
    )
    df_test_mixed.to_csv(files_path / "edos_labelled_mixed.csv", index=False)

    return "Los datasets de consenso han sido creados"


def get_model_path(dataset: DatasetEnum, model_name: ModelsEnum):
    """Get the pretrained model for sexism classification and tokenizer or train a new one"""
    model_path = os.path.join(
        settings.BIAS_MODEL_PATH,
        dataset.model_folder_path,
        model_name.value.split("/")[-1],
    )

    if os.path.exists(model_path):
        return model_path
    else:
        raise ValueError("Modelo no encontrado")


def predict_sexism_batch(model_path: str, texts: list[str]) -> list[dict[str, float]]:
    """
    Carga un modelo de clasificación desde model_path y clasifica una lista de textos.
    Retorna una lista de diccionarios, donde cada diccionario tiene al menos:
        {"label": <str>, "score": <float>}
    """
    # Cargar el tokenizer y el modelo entrenado
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Crear el pipeline de clasificación
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    # Clasificar la lista de textos (si texts es muy grande, hace batching automáticamente)
    results = classifier(texts)

    return results


def sexism_evaluator_test_samples(eval_path, model_path, csv_path):
    # Cargar el tokenizer y el modelo entrenado
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Función de tokenización (igual que en entrenamiento)
    def tokenize_function(example):
        return tokenizer(
            example["text"], padding="max_length", truncation=True, max_length=128
        )

    # Cargar las métricas
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "precision": precision.compute(predictions=preds, references=labels)[
                "precision"
            ],
            "recall": recall.compute(predictions=preds, references=labels)["recall"],
            "f1": f1.compute(predictions=preds, references=labels)["f1"],
        }

    # Iniciar el Trainer
    eval_args = TrainingArguments(
        output_dir="eval_output",
        per_device_eval_batch_size=16,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    print(f"\nEvaluando en {csv_path}...")
    df_test = pd.read_csv(csv_path)

    # Renombra la columna label_sexist por label
    if "label_sexist" in df_test.columns:
        df_test = df_test.rename(columns={"label_sexist": "label"})

    if "split" in df_test.columns:
        # Si hay columna split, nos quedamos solo con el test
        df_test = df_test[df_test["split"] == "test"]

    if df_test["label"].dtype == object:  # son strings
        # Ej: mapeo "sexist"->1, "not sexist"->0
        df_test["label"] = df_test["label"].map({"not sexist": 0, "sexist": 1})

    # Convertimos a Dataset
    dataset_test = Dataset.from_pandas(df_test)

    # Tokenizamos
    dataset_test = dataset_test.map(tokenize_function, batched=True)

    # Removemos columnas que sobran (por ejemplo, "text")
    dataset_test = dataset_test.remove_columns(["text"])

    # Llamamos a evaluate
    metrics = trainer.evaluate(dataset_test)
    print(f"Resultados en {csv_path}: {metrics}")

    # Si no existe la carpeta, se crea
    if not os.path.exists(os.path.dirname(eval_path)):
        os.makedirs(os.path.dirname(eval_path))

    # Guarda en models/evaluations los resultados según el modelo (eval_path)
    #  y el csv, lo añade, no borra
    with open(eval_path, "a") as f:
        f.write(f"Resultados en {csv_path}: {metrics}\n")

    return metrics
