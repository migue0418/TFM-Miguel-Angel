from app.core.config import files_path
import pandas as pd

from app.utils.preprocessing import clean_text_from_edos


def generate_4_sexism_labels_dataset():
    # Load the dataset
    csv_path = files_path / "edos_labelled_individual_annotations.csv"

    # Read it with pandas
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

    grouped["sexism_grade"] = grouped["annot_labels"].apply(classify_4_labels)

    # Ahora grouped tiene columnas:
    #  - rewire_id
    #  - annot_labels (lista de 3 strings)
    #  - sexism_grade ("sexist", "sexist (high confidence)",
    #                  "sexist (low confidence)" o "not sexist")

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

    # Si quieres, revisa cuántos rewire_id hay en cada categoría
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
