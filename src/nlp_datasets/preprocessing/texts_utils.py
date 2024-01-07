import emoji
import re


def clean_text(
    text: str,
    remove_html: bool = True,
    remove_emoji: bool = True,
    remove_url: bool = True,
    remove_tabs_new_line: bool = True,
    lower_text: bool = True,
) -> str:
    """Basic function to clean the text

    Args:
        text (str): the input text
        remove_html (bool, optional): remove html part of the text. Defaults to True.
        remove_emoji (bool, optional): remove emoji from the text. Defaults to True.
        remove_url (bool, optional): remove the entire url from the text. Defaults to True.
        remove_tabs_new_line (bool, optional): remove tabs, newlines and multiple spaces. Defaults to True.
        lower_text (bool, optional): lower the text. Defaults to True.

    Returns:
        str: the cleaned text
    """

    if remove_html:
        html = re.compile(r"<.*?>")
        text = html.sub(r"", text)
    if remove_emoji:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r"", text)
        text = emoji.demojize(text)
    if remove_url:
        url = re.compile(r"https?://\S+|www\.\S+")
        text = url.sub(r"", text)
    if remove_tabs_new_line:
        text = re.sub(
            "[ \t\n]+", " ", text
        )  # Remove tabs, newlines and multiple spaces
    if lower_text:
        text = text.lower()

    return text.strip()
