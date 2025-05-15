from google.cloud import translate_v2 as translate


def translate_text(input_text: str, source_language: str, target_language: str) -> str:
    """
    Translate text from one language to another using Google Cloud Translation API.
    :param input_text: text to translate
    :param source_language: language of the text to translate (e.g., "en" for English, "es" for Spanish)
    :param target_language: language to translate the text to (e.g., "fr" for French)
    :return: Text translated, or the original text if translation fails.
    """
    if not input_text:
        return ""
    # init translator client
    translate_client = translate.Client()
    # execute translation
    result = translate_client.translate(
        input_text,
        target_language=target_language,
        source_language=source_language
    )
    # get result
    translated_text = result['translatedText']
    #  return translated text
    return str(translated_text)
