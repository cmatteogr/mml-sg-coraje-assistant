from typing import Tuple
from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step
from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher


@step
def crawl_links(links: list[Tuple[str, str]]) -> Annotated[list[str], "crawled_links"]:
    dispatcher = CrawlerDispatcher.build().register_transcription().register_ml_book()

    logger.info(f"Starting to crawl {len(links)} link(s).")

    metadata = {}
    successfull_crawls = 0
    for domain, link in tqdm(links):
        successfull_crawl, crawled_domain = _crawl_link(dispatcher, link, domain)
        successfull_crawls += successfull_crawl

        metadata = _add_to_metadata(metadata, crawled_domain, successfull_crawl)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="crawled_links", metadata=metadata)

    logger.info(f"Successfully crawled {successfull_crawls} / {len(links)} links.")

    return links


def _crawl_link(dispatcher: CrawlerDispatcher, link: str, domain: str) -> tuple[bool, str]:
    crawler = dispatcher.get_crawler(domain)
    crawler_domain = domain

    try:
        crawler.extract(link=link)

        return (True, crawler_domain)
    except Exception as e:
        logger.error(f"An error occurred while crowling: {e!s}")

        return (False, crawler_domain)


def _add_to_metadata(metadata: dict, domain: str, successfull_crawl: bool) -> dict:
    if domain not in metadata:
        metadata[domain] = {}
    metadata[domain]["successful"] = metadata.get(domain, {}).get("successful", 0) + successfull_crawl
    metadata[domain]["total"] = metadata.get(domain, {}).get("total", 0) + 1

    return metadata


"""links = [
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
crawl_links(links)"""