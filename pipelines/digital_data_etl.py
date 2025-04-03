from zenml import pipeline

@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> str:
    last_step = crawl_links(user=user, links=links)

    return last_step.invocation_id
