import base64
import hashlib
import re
from datetime import datetime

from Crypto.Cipher import AES

AVAILABLE_MODEL = [""]
AVAILABLE_SOURCE = ["system", "local", "s3", "oss", "ftp", "email", "feishu", "url","github"]


def extract_bot_and_chat_id(path: str):
    match = re.match(
        r"/api/v1/bots/(?P<bot_id>\w+)/chats/(?P<chat_id>\w+)/connect$",
        path,
    )
    if match:
        return match.group("bot_id"), match.group("chat_id")
    else:
        raise ValueError(f"Invalid path format: {path}")


def extract_web_bot_and_chat_id(path: str):
    match = re.match(
        r"/api/v1/bots/(?P<bot_id>\w+)/web-chats/(?P<chat_id>\w+)/connect$",
        path,
    )
    if match:
        return match.group("bot_id"), match.group("chat_id")
    else:
        raise ValueError(f"Invalid path format: {path}")


def extract_collection_and_chat_id(path: str):
    match = re.match(
        r"/api/v1/collections/(?P<collection_id>\w+)/chats/(?P<chat_id>\w+)/connect$",
        path,
    )
    if match:
        return match.group("collection_id"), match.group("chat_id")
    else:
        raise ValueError(f"Invalid path format: {path}")


def extract_chat_id(path: str):
    match = re.match(r"/api/v1/bot/(?P<chat_id>\w+)/connect$", path)
    if match:
        return match.group("chat_id")
    else:
        raise ValueError(f"Invalid path format: {path}")


def extract_database(path: str, db_type):
    if db_type in ["mysql", "postgresql"]:
        match = re.match(r"database=(?P<database>\w+)", path)
        if match:
            return match.group("database")
        else:
            raise ValueError(f"Invalid path format: {path}")
    else:
        return None


def extract_code_chat(path: str):
    match = re.match(r"/api/v1/code/(?P<chat_id>\w+)/connect$", path)
    if match:
        return match.group("chat_id")
    else:
        raise ValueError(f"Invalid path format: {path}")


def now_unix_milliseconds():
    return int(datetime.utcnow().timestamp() * 1e3)


def generate_fulltext_index_name(collection_id) -> str:
    return str(collection_id)


def generate_vector_db_collection_name(collection_id) -> str:
    return str(collection_id)

def generate_qa_vector_db_collection_name(collection) -> str:
    return str(collection) + "-qa"

def fix_path_name(path) -> str:
    return str(path).replace("|", "-")


class AESCipher(object):
    def __init__(self, key):
        self.bs = AES.block_size
        self.key = hashlib.sha256(AESCipher.str_to_bytes(key)).digest()

    @staticmethod
    def str_to_bytes(data):
        u_type = type(b"".decode('utf8'))
        if isinstance(data, u_type):
            return data.encode('utf8')
        return data

    @staticmethod
    def _unpadding(s):
        return s[:-ord(s[len(s) - 1:])]

    def decrypt(self, enc):
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpadding(cipher.decrypt(enc[AES.block_size:]))

    def decrypt_string(self, enc):
        enc = base64.b64decode(enc)
        return self.decrypt(enc).decode('utf8')


class Stacks:
    """
    An array of stacks for local document embedding
    the array index is the docx title level
    the every stack store the level contents
    """

    def __init__(self):
        self.stacks = [[]]  # [] is a placeholder

    def push(self, level: int, value: str):
        """
        push the string to the stack at level
        :param level: the level of the stack we want to push
        :param value: content of the string
        """
        while level >= len(self.stacks):
            self.stacks.append([])
        self.stacks[level].append(value)

    def pop(self, level: int):
        """
        pop from the stack at level
        :param level: the level of the stack we want to pop
        :return: the pop elelment
        """
        if level >= len(self.stacks):
            return None
        if len(self.stacks[level]) == 0:
            return None
        return self.stacks[level].pop()

    def package_content(self, level: int):
        """
        package the stack contents to a trunk from  0 ~ level
        :param level: the deepest level to package
        :return: content string
        """
        res = ""
        for i in range(0, level + 1):
            for j in range(0, len(self.stacks[i])):
                if j == 0:
                    res += "\n"      # add "\n" for different level title
                res += self.stacks[i][j]

        return res

    def count_contents(self, level: int):
        """
        count the contexts for level
        :param level: the stack level we want to count
        :return: the total length of content at level
        """
        res = 0
        for i in range(0, len(self.stacks[level])):
            res += len(self.stacks[level][i])
        return res

    def remove(self, level):
        """
        remove the stacks contents from level to the deepest level
        :param level: begin level
        """
        for i in range(level, len(self.stacks)):
            while len(self.stacks[i]) > 0:
                self.pop(i)

    def get_title(self, level):
        """
        get the title
        :param level: the level of the title
        :return: the content of title
        """
        return self.stacks[level][0]
