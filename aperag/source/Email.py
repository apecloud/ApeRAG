import datetime
import imaplib
import logging
import poplib
import re
import tempfile
from email import message_from_bytes, parser
from email.header import decode_header
from typing import Any, Dict, Iterator

from bs4 import BeautifulSoup

from aperag.source.base import CustomSourceInitializationError, LocalDocument, RemoteDocument, Source

logger = logging.getLogger(__name__)


def download_email_body_to_temp_file(mail_conn, email_index, name, protocol):
    if protocol.upper() == "IMAP":
        mail_conn.select("INBOX")
        _, data = mail_conn.fetch(email_index, '(RFC822)')
        message_content = data[0][1]
    else:
        _, message_lines, _ = mail_conn.retr(email_index)
        message_content = b"\r\n".join(message_lines)

    prefix, plain_text = get_email_plain_text(message_content, name)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=prefix,
        delete=False,
        suffix=".txt",
    )
    temp_file.write(plain_text.encode("utf-8"))
    temp_file.close()
    return temp_file.name


# get the plain text in email
def get_email_plain_text(message_content, name):
    message = message_from_bytes(message_content)
    body = ""
    for part in message.walk():
        content_type = part.get_content_type()
        if content_type == 'text/plain' or content_type == 'text/html':
            content = part.get_payload(decode=True)
            charset = part.get_charset()
            if charset is None:
                content_type = part.get('Content-Type', '').lower()
                position = content_type.find('charset=')
                if position >= 0:
                    charset = content_type[position + 8:].strip()
                    if charset.find("utf-8") != -1:
                        charset = '"utf-8"'
            if charset:
                text_content = content.decode(charset)
                body += text_content
    plain_text = extract_plain_text_from_email_body(body)
    prefix = name.strip("/").replace("/", "--")

    # when all email context is pure html without text, fill with its title
    if plain_text == '':
        plain_text = name
    return prefix, plain_text


def extract_plain_text_from_email_body(body):
    soup = BeautifulSoup(body, "html.parser")
    for style_tag in soup.find_all("style"):
        style_tag.decompose()
    for script_tag in soup.find_all("meta"):
        script_tag.decompose()
    for meta_tag in soup.find_all("html"):
        meta_tag.decompose()
    return soup.get_text()


def decode_msg_header(header):
    value, charset = decode_header(header)[0]
    if charset:
        value = value.decode(charset)
    if isinstance(value, bytes):
        value.decode()
        value = value.decode()
    return value


# check if text contain chinese character
def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')  # search chinese character
    result = re.search(pattern, text)
    return result is not None


# check if chinese/english email is spam
def check_spam(title: str, body: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    chinese = contains_chinese(title)
    if chinese:
        model_name = "paulkm/chinese_spam_detect"
        max_length = 512
    else:
        model_name = "mrm8488/bert-tiny-finetuned-enron-spam-detection"
        max_length = 512
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(body, truncation=True, max_length=max_length, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    return pred.item() > 0


class EmailSource(Source):


    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        self.server = ctx["pop_server"]
        self.port = ctx["port"]
        self.email_address = ctx["email_address"]
        self.email_password = ctx["email_password"]
        self.protocol = ""
        self.detect_spam = ctx.get("detect_spam", False)
        self.conn = self._connect_to_mail_server()
        self.email_num = 0

    def _connect_to_mail_server(self):
        # Check if email format is valid
        if not re.match(r"[^@]+@[^@]+\.[^@]+", self.email_address):
            raise CustomSourceInitializationError("Invalid email format")

        # Try to connect using IMAP
        try:
            timeout = 5
            if self.port == 143:
                conn = imaplib.IMAP4(self.server, self.port, timeout=timeout)
            else:
                conn = imaplib.IMAP4_SSL(self.server, self.port, timeout=timeout)
            conn.login(self.email_address, self.email_password)
            self.protocol = "IMAP"
            return conn
        except Exception as imap_error:
            # IMAP connection failed, now try POP
            try:
                if self.port == 110:
                    conn = poplib.POP3(self.server, self.port, timeout=timeout)
                else:
                    conn = poplib.POP3_SSL(self.server, self.port, timeout=timeout)
                conn.user(self.email_address)
                conn.pass_(self.email_password)
                self.protocol = "POP"
                return conn
            except Exception as pop_error:
                raise CustomSourceInitializationError(
                    f"Failed to connect to mail server,IMAP for: {imap_error}, then POP for: {pop_error}")

    def scan_documents(self) -> Iterator[RemoteDocument]:
        try:
            if self.protocol.upper() == "IMAP":
                self.conn.select("INBOX")
                _, data = self.conn.search(None, "ALL")
                email_nums = data[0].split()
                self.email_num = len(email_nums)
            else:
                _, email_list = self.conn.list()
                self.email_num = len(email_list)
            for i in range(self.email_num):
                try:
                    if self.protocol.upper() == "IMAP":
                        _, data = self.conn.fetch(email_nums[i], '(RFC822)')
                        msg_lines = data[0][1]
                        octets = len(msg_lines)
                    else:
                        _, msg_lines, octets = self.conn.retr(i + 1)
                        msg_lines = b"\r\n".join(msg_lines)

                    msg_lines_undecoded = msg_lines
                    msg_lines_to_str = msg_lines_undecoded.decode("utf8", "ignore")
                    message_object = parser.Parser().parsestr(msg_lines_to_str)

                    msg_header = message_object["Subject"]
                    decoded_subject = decode_msg_header(msg_header)
                    order_and_name = str(i + 1) + '_' + decoded_subject + ".txt"

                    # check if spam,if it is spam, jump to next email
                    if self.detect_spam:
                        _, message_content = get_email_plain_text(msg_lines_undecoded, decoded_subject)
                        is_spam = check_spam(decoded_subject, message_content)
                        if is_spam:
                            logger.info(f"email {decoded_subject} is detected to be spam")
                            continue
                        logger.info(f"email {decoded_subject} is detected to be ham")

                    document = RemoteDocument(
                        name=order_and_name,
                        size=octets,
                        metadata={
                            # TODO: use the email received time as the modified time
                            "modified_time": datetime.datetime.now(),
                        }
                    )
                    yield document
                except Exception as e:
                    logger.error(f"scan_email_documents {e}")
                    raise e
        except Exception as e:
            logger.error(f"Error in scan_documents: {e}")
            raise e

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        under_line = name.find('_')
        order = name[:under_line]
        temp_file_path = download_email_body_to_temp_file(
            self.conn, order, name, self.protocol
        )
        metadata["name"] = name
        return LocalDocument(name=name, path=temp_file_path, metadata=metadata)

    def close(self):
        self.conn.quit()

    def sync_enabled(self):
        return True
