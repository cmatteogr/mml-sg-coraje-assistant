from typing import Tuple

from zenml import pipeline
from steps.etl import crawl_links


@pipeline
def digital_data_etl(links: list[Tuple[str, str]]) -> str:
    last_step = crawl_links(links=links)

    return last_step.invocation_id


links = [
    ('ml_books', 'A Practical Outlier Detection Approach for Mixed Attibute Data.pdf'),
    ('ml_books', 'Bagging Predictors.pdf'),
    ('ml_books', 'Daily Dose Of Data Science Full Archive.pdf'),
    ('ml_books', 'Data Preparation for Machine Learning - Jason Brownlee.pdf'),
    ('ml_books', 'Deep Generative Modeling.pdf'),
    ('ml_books', 'Deep Learning Book - Ian Yoshua Aaron.pdf'),
    ('ml_books', 'Deep Learning for Time Series Forecasting - Predict the Future with MLPs, CNNs and LSTMs in Python by Jason Brownlee (z-lib.org).pdf'),
    ('ml_books', 'Deep Learning on Graphs - Yao Jilliang.pdf'),
    ('ml_books', 'Energy-Based Anomaly Detection for Mixed Data.pdf'),
    ('transcription', '2. Exploración de Modelos de ML y Exploración de Datos (2024-02-28 19_14 GMT-5).txt'),
    ('transcription', '3. Análisis de Datos y Selección de Variables para Modelado (2024-03-06 19_08 GMT-5).txt'),
    ('transcription', '4. Construcción del Modelo de Predicción - Supervised Learning (2024-03-13 19_07 GMT-5).txt'),
    ('transcription', '5. Supervised Learning - Optimización del Modelo (2024-04-10 19_11 GMT-5).txt')
]
digital_data_etl(links)