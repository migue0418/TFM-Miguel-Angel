from app.enums.datasets_enums import DatasetEnum
from app.enums.models_enums import ModelsEnum
from app.core.config import files_path
from app.core.config import settings
from app.utils.preprocessing import clean_text_from_edos
import pandas as pd
import os
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments


def generate_4_sexism_labels_dataset():
    # Cargar el dataset con pandas
    csv_path = files_path / "edos_labelled_individual_annotations.csv"
    df = pd.read_csv(csv_path)

    # Agrupar por rewire_id y recolectar las etiquetas en una lista
    grouped = (
        df.groupby("rewire_id")["label_sexist"]
        .apply(list)
        .reset_index(name="annot_labels")
    )

    # Clasificar cada rewire_id en base a las 4 anotaciones
    def classify_4_labels(label_list):
        # Hacemos un count de sexist
        count_sexist = label_list.count("sexist")
        # Si tiene 3 es "sexist"
        if count_sexist == 3:
            return "sexist"
        # Si tiene 2 es "sexist (high confidence)"
        if count_sexist == 2:
            return "sexist (high confidence)"
        # Si tiene 1 es "sexist (low confidence)"
        if count_sexist == 1:
            return "sexist (low confidence)"
        # Si tiene 0 es "not sexist"
        return "not sexist"

    # Añadimos el grado de sexismo
    grouped["sexism_grade"] = grouped["annot_labels"].apply(classify_4_labels)

    # Carga el dataset de aggregated y obtiene la etiqueta de ahí uniéndolo con grouped
    df_aggregated = pd.read_csv(files_path / "edos_labelled_aggregated.csv")

    # Hacemos una limpieza suave del dataset
    df_aggregated["text"] = df_aggregated["text"].apply(clean_text_from_edos)

    # Mergeamos con el dataset de aggregated porque ya los tiene agrupados
    df_merged = pd.merge(
        df_aggregated,
        grouped[["rewire_id", "sexism_grade"]],
        on="rewire_id",
        how="left",
    )

    # Guardamos el nuevo CSV con las 4 etiquetas
    df_merged.to_csv(files_path / "edos_labelled_4_sexism_grade.csv", index=False)

    # Obtenemos un dataset reducido para entrenar más rápido
    data_sexist = df_merged[df_merged["sexism_grade"] == "sexist"].sample(
        n=1250, random_state=42
    )
    data_not_sexist = df_merged[df_merged["sexism_grade"] == "not sexist"].sample(
        n=1250, random_state=42
    )
    data_high_sexist = df_merged[
        df_merged["sexism_grade"] == "sexist (high confidence)"
    ].sample(n=1250, random_state=42)
    data_low_sexist = df_merged[
        df_merged["sexism_grade"] == "sexist (low confidence)"
    ].sample(n=1250, random_state=42)

    # Concatenamos los dataframes
    data_reduced = pd.concat(
        [data_sexist, data_not_sexist, data_high_sexist, data_low_sexist]
    )

    # Hacemos un shuffle de los datos
    data_reduced = data_reduced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Guardamos el dataset reducido
    data_reduced.to_csv(
        files_path / "edos_labelled_4_sexism_grade_reduced.csv", index=False
    )

    # Obtenemos sólo las filas que pertenecen a test
    df_test = df_merged[df_merged["split"] == "test"]

    # Finalmente separamos en 3 dataframes (por unanimidad vs. mixto):
    df_test_all_sexist = df_test[df_test["sexism_grade"] == "sexist"].copy()
    df_test_all_not_sexist = df_test[df_test["sexism_grade"] == "not sexist"].copy()
    df_test_high_confidence = df_test[
        df_test["sexism_grade"] == "sexist (high confidence)"
    ].copy()
    df_test_low_confidence = df_test[
        df_test["sexism_grade"] == "sexist (low confidence)"
    ].copy()

    # Agrupar por rewire_id para evitar repetidos
    df_test_all_sexist = df_test_all_sexist.groupby("rewire_id").first().reset_index()
    df_test_all_not_sexist = (
        df_test_all_not_sexist.groupby("rewire_id").first().reset_index()
    )
    df_test_high_confidence = (
        df_test_high_confidence.groupby("rewire_id").first().reset_index()
    )
    df_test_low_confidence = (
        df_test_low_confidence.groupby("rewire_id").first().reset_index()
    )

    # Imprimimos el número de IDs en cada categoría
    print("All sexist:", df_test_all_sexist["rewire_id"].nunique())
    print("All not sexist:", df_test_all_not_sexist["rewire_id"].nunique())
    print("Sexist (high confidence):", df_test_high_confidence["rewire_id"].nunique())
    print("Sexist (low confidence):", df_test_low_confidence["rewire_id"].nunique())

    # Guarda los datasets como CSVs separados
    df_test_all_sexist.to_csv(files_path / "edos_4_grades_sexist.csv", index=False)
    df_test_all_not_sexist.to_csv(
        files_path / "edos_4_grades_not_sexist.csv", index=False
    )
    df_test_high_confidence.to_csv(
        files_path / "edos_4_grades_sexist_high.csv", index=False
    )
    df_test_low_confidence.to_csv(
        files_path / "edos_4_grades_sexist_low.csv", index=False
    )

    return "Los datasets de consenso han sido creados"


def train_model_4_labels(dataset: DatasetEnum, model_name: ModelsEnum):
    """
    Entrena un modelo para detección de sexismo (4 clases: sexist / not sexist
      / sexis (high confidence) / sexist (low confidence)).

    - Si es un dataset de EDOS, se asume un CSV con columnas:
      ['text', 'label_sexist', 'split', ...] y usa train/dev/test de la columna 'split'.
    """
    # Obtener ruta donde se guardará el modelo
    model_path = os.path.join(
        settings.BIAS_MODEL_PATH,
        dataset.model_folder_path,
        model_name.value.split("/")[-1],
    )

    # Cargar el dataset desde CSV
    df = pd.read_csv(dataset.csv_path)

    # Asegurar que las columnas 'text', 'label_sexist' y 'split' existen
    if not all(col in df.columns for col in ["text", "sexism_grade", "split"]):
        raise ValueError(
            "El CSV de EDOS debe contener las columnas 'text', 'sexism_grade' y 'split'."
        )

    # Crear splits a partir de la columna 'split'
    df_train = df[df["split"] == "train"]
    df_dev = df[df["split"] == "dev"]
    df_test = df[df["split"] == "test"]

    # Renombrar 'text' -> 'comment' y 'sexism_grade' -> 'label' (por convención en HF)
    df_train = df_train.rename(columns={"text": "comment", "sexism_grade": "label"})
    df_dev = df_dev.rename(columns={"text": "comment", "sexism_grade": "label"})
    df_test = df_test.rename(columns={"text": "comment", "sexism_grade": "label"})

    # Convertir cada split en Dataset
    dataset_train = Dataset.from_pandas(df_train)
    dataset_dev = Dataset.from_pandas(df_dev)
    dataset_test = Dataset.from_pandas(df_test)

    # Convertir la columna "label" a un tipo ClassLabel (string -> int) de forma automática
    dataset_train = dataset_train.class_encode_column("label")
    dataset_dev = dataset_dev.class_encode_column("label")
    if dataset_test is not None:
        dataset_test = dataset_test.class_encode_column("label")

    # Cargar el tokenizador y modelo base
    tokenizer = AutoTokenizer.from_pretrained(model_name.value)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name.value, num_labels=4
    )

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
            "precision": precision_metric.compute(
                predictions=preds, references=labels, average="macro"
            )["precision"],
            "recall": recall_metric.compute(
                predictions=preds, references=labels, average="macro"
            )["recall"],
            "f1": f1_metric.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"],
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


"""
Modelo y tokenizador guardados en app/models/bias_classifier\edos_4_sexism_reduced\ModernBERT-base
label2id: {'not sexist': 0, 'sexist': 1, 'sexist (high confidence)': 2, 'sexist (low confidence)': 3}
id2label: {0: 'not sexist', 1: 'sexist', 2: 'sexist (high confidence)', 3: 'sexist (low confidence)'}



Valores de test con el de 4 etiqutas y el de 2 en el evaluate. Para ver cuanto te aciertas y equivocas en cada uno.

Poner que modelos probamos y que funcionó, darle una vuelta a la presentación.
Learning rate
Batch size
Steps

Probar otro modelo aparte de bert
"""
