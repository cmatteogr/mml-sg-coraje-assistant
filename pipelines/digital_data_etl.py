from typing import Tuple

from steps.etl import crawl_links

def digital_data_etl(links: list[Tuple[str, str]]) -> str:
    last_step = crawl_links(links=links)

    # return last_step.invocation_id


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
    ('ml_books', 'Feature Engineering for Machine Learning.pdf'),
    ('ml_books', 'Finite Bivariate and Multivariate Beta Mixture Models.pdf'),
    ('ml_books', 'Fourier Features in Reinforcement Learning with Neural Networks - Brellmann - Filliat - Frehse.pdf'),
    ('ml_books', 'Game Theory and Machine Learning for Cyber Security.pdf'),
    ('ml_books', 'Generative Adversarial Networks Bridging Art and Machine Intelligence.pdf'),
    ('ml_books', 'Generative Adversarial Networks with Python Deep Learning Generative Models for Image Synthesis and Image Translation by Jason Brownlee (z-lib.org).pdf'),
    ('ml_books', 'Graph Representation Learning - William Hamilton.pdf'),
    ('ml_books', 'Imbalanced Classification with Python Choose Better Metrics, Balance Skewed Classes, and Apply Cost-Sensitive Learning by Jason Brownlee (z-lib.org) (1).pdf'),
    ('ml_books', 'Information Theory, Inference, and Learning Algorithms - David MacKay.pdf'),
    ('ml_books', 'Introduction to Statistical Learning - Python.pdf'),
    ('ml_books', 'Long Short-Term Memory Networks With Python Develop Sequence Prediction Models With Deep Learning by Jason Brownlee (z-lib.org).pdf'),
    ('ml_books', 'Machine Learning Mastery with Python Understand Your Data, Create Accurate Models and Work Projects End-To-End by Jason Brownlee (z-lib.org).pdf'),
    ('ml_books', 'ML Machine Learning-A Probabilistic Perspective.pdf'),
    ('ml_books', 'Modern Graph Theory - Bela Bollobas.pdf'),
    ('ml_books', 'Neural networks for abstraction and reasoning - Bober and Banerjee.pdf'),
    ('ml_books', 'Pasting Predictors.pdf'),
    ('ml_books', 'Probabilistic Graphical Models - Principles and Techniques.pdf'),
    ('ml_books', 'ProcessAutomation_Camunda.pdf'),
    ('ml_books', 'Reinforcing Learning Introduction.pdf'),
    ('ml_books', 'SQL for Data Scientists a Beginners Guide for Building Datasets for Analysis.pdf'),
    ('ml_books', 'Statistical Methods for Machine Learning Discover How to Transform Data into Knowledge with Python by Jason Brownlee (z-lib.org) (1).pdf'),
    ('ml_books', 'Steven G. Krantz - Techniques of Problem Solving-American Mathematical Society (1997).pdf'),
    ('ml_books', 'The Art of Desception -  by Kevin Mitnick.pdf'),
    ('ml_books', 'The Art of Invisibility - The Worlds Most Famous Hacker Teaches You How to Be Safe in the Age of Big Brother and Big Data by Kevin Mitnick.pdf'),
    ('ml_books', 'The Elements of Statistical Learning - Trevor Rober Jerome.pdf'),
    ('ml_books', 'Understanding Deep Learning.pdf'),

    ('transcription', '2. Exploración de Modelos de ML y Exploración de Datos (2024-02-28 19_14 GMT-5).txt'),
    ('transcription', '3. Análisis de Datos y Selección de Variables para Modelado (2024-03-06 19_08 GMT-5).txt'),
    ('transcription', '4. Construcción del Modelo de Predicción - Supervised Learning (2024-03-13 19_07 GMT-5).txt'),
    ('transcription', '5. Supervised Learning - Optimización del Modelo (2024-04-10 19_11 GMT-5).txt'),
    ('transcription', '6. Implementación de la Detección de Anomalías (2024-04-17 19_09 GMT-5).txt'),
    ('transcription', '7. Evaluación del Modelo y Resultados de la Detección de Anomalías (2024-04-24 19_09 GMT-5).txt'),
    ('transcription', '8. Preparación para el Despliegue del Modelo (2024-05-02 19_09 GMT-5).txt'),
    ('transcription', '9. Despliegue del Modelo y Monitoreo en Producción (2024-05-08 19_12 GMT-5).txt'),
    ('transcription', '10. Revisión Final y Lecciones Aprendidas del Proyecto de Predicción de Precios de Carros (2024-05-15 19_11 GMT-5).txt'),
    ('transcription', '11. Introducción al Uso de Autoencoders en Detección de Anomalías (2024-05-22 19_34 GMT-5).txt'),
    ('transcription', '12. Implementación de Autoencoders para Detección de Anomalías en Datos de Carros (2024-05-29 19_11 GMT-5).txt'),
    ('transcription', '13. Evaluación del Desempeño de Autoencoders en la Detección de Anomalías (2024-06-05 19_10 GMT-5).txt'),
    ('transcription', '14. Optimización de Autoencoders para Mejorar la Detección de Anomalías (2024-06-12 19_10 GMT-5).txt'),
    ('transcription', '15. Integración de Autoencoders en el Pipeline de Predicción de Precios de Carros (2024-06-18 19_12 GMT-5).txt'),
    ('transcription', '16. Despliegue de Autoencoders para la Detección de Anomalías en Producción (2024-07-03 19_07 GMT-5).txt'),
    ('transcription', '17. Monitoreo y Mantenimiento del Sistema de Detección de Anomalías (2024-07-24 19_29 GMT-5).txt'),
    ('transcription', '18. Revisión Final y Lecciones Aprendidas sobre Autoencoders para Detección de Anomalías (2024-08-07 19_09 GMT-5).txt'),
    ('transcription', '19. Bases de RAGs para construir un asistente virtual (2024-08-14 19_09 GMT-5).txt'),
    ('transcription', '20. Introducción a Grafos Machine Learning (GNN) (2024-08-21 19_09 GMT-5).txt'),
    ('transcription', '21. Principios base de Grafos en Machine Learning (2024-08-28 19_09 GMT-5).txt'),
    ('transcription', '22. Deep Learning en Grafos Introduccion (2024-09-04 19_09 GMT-5).txt'),
    ('transcription', '23. Graph Embeddings Introduccion y  Attack-Defend Game (2024-09-11 19_10 GMT-5)).txt'),
    ('transcription', '24. Understanding Graph Embedding and Profiling Cyberattacks (2024_09_18 18_59 COT).txt'),
    ('transcription', '25. Graph Embeddings Introduction (2024-10-02 19_09 GMT-5).txt'),
    ('transcription', '26. Community Detection Introduction (2024-10-09 19_09 GMT-5).txt'),

    ('github_code', 'cmatteogr/cars_anomaly_detection_autoencoder'),
    ('github_code', 'cmatteogr/cars_ml_project'),
    ('github_code', 'cmatteogr/cars_model_deployment'),
    ('github_code', 'cmatteogr/cars_scrapy'),
    ('github_code', 'cmatteogr/generative-ml-malicious-traffic-wolf-in-sheeps-clothes'),
    ('github_code', 'cmatteogr/graph_attack_defend_games_ml'),
    ('github_code', 'cmatteogr/medellin_ai_autoencoder_anomaly_detection'),
    ('github_code', 'cmatteogr/mml-sg-coraje-assistant'),
]

digital_data_etl(links)