from zenml import pipeline

from steps import feature_engineering as fe_steps


@pipeline(enable_cache=False)
def feature_engineering(ml_book_names: list[str], transcription_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(ml_book_names, transcription_names, after=wait_for)

    # translate documents
    # NOTE: Only needed for the Transcriptions
    translated_documents = fe_steps.translate_documents(raw_documents)


    # clean documents
    cleaned_documents = fe_steps.clean_documents(raw_documents)
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)

    # get embedded documents
    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

    return [last_step_1.invocation_id, last_step_2.invocation_id]


ml_book_names = [
    'A Practical Outlier Detection Approach for Mixed Attibute Data',
    'Bagging Predictors',
    'Daily Dose Of Data Science Full Archive',
    'Data Preparation for Machine Learning - Jason Brownlee',
    'Deep Generative Modeling',
    'Deep Learning Book - Ian Yoshua Aaron'
]

transcription_names = [
    '2. Exploración de Modelos de ML y Exploración de Datos (2024-02-28 19_14 GMT-5)',
    '3. Análisis de Datos y Selección de Variables para Modelado (2024-03-06 19_08 GMT-5)',
    '4. Construcción del Modelo de Predicción - Supervised Learning (2024-03-13 19_07 GMT-5)',
    '5. Supervised Learning - Optimización del Modelo (2024-04-10 19_11 GMT-5)'
]
feature_engineering(ml_book_names, transcription_names)