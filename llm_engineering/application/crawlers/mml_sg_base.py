from loguru import logger
from .base import BaseLocalCrawler
import os
from llm_engineering.domain.documents import MMLSGBaseDocument
from pathlib import Path



class MMLSGBaseCrawler(BaseLocalCrawler):
    model = MMLSGBaseDocument

    def extract(self, link: str, **kwargs) -> None:
        mml_sg_base_name = Path(os.path.basename(link)).stem
        logger.info(f"Starting scrapping data for MML-SG base: {link}")

        old_model = self.model.find(name=mml_sg_base_name)
        if old_model is not None:
            logger.info(f"MML-SG base already exists in the database: {link}")
            return

        # init destination and source paths
        destination_path = os.path.join(self.destination_folder, 'mml_sg_base', link)
        # check if source file exist
        if not os.path.exists(destination_path):
            raise FileNotFoundError(f"{destination_path} does not exist")

        mml_sg_base = self._extract_mml_sg_base(destination_path)
        logger.info(f"Found {len(mml_sg_base)} MML-SG base for: {link}")
        # save new book
        instance = self.model(platform="mml_sg_base", content=mml_sg_base, name=mml_sg_base_name,
                              filepath=destination_path)
        instance.save()

        logger.info(f"Finished scrapping data for MML-SG base: {link}")

    def _extract_mml_sg_base(self, filepath: str) -> str:
        """
        Extracts transcription text.
        """
        with open(filepath, "r") as file:
            content = file.read()
        return content
