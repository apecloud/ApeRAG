from pathlib import Path
from typing import Dict, List, Optional

import requests
from llama_index.readers.base import BaseReader
from llama_index.schema import Document

import config.settings as settings

WHISPER_SERVER_URL = settings.WHISPER_HOST


class ComposeAudioReader(BaseReader):
    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        text = handle_audio(file)
        return [
            Document(
                text="%s\n" % (text),
                metadata=metadata or {},
            )
        ]


def handle_audio(path):
    params = {
        'encode': 'true',
        'task': 'transcribe',
        'vad_filter': 'true',
        'word_timestamps': 'true',
        'output': 'txt'
    }

    files = {'audio_file': open(path, 'rb')}

    headers = {
        'Accept': 'application/json',
    }

    response = requests.post(WHISPER_SERVER_URL + '/asr', params=params, files=files, headers=headers)
    return response.text
