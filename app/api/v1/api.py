import librosa
import os
import tempfile
from fastapi import APIRouter, UploadFile, File

from app.core.config import local_settings
from app.schemas import ResponseModel
from app.services.demo import demo
from app.services.process_wav import convert_to_16k_mono_librosa

import sys
sys.path.append("/app/SSL_Anti-spoofing")
from  Multikol_inference import *

class MultikolServiceSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, local_settings):
        self.service = multikol_service(local_settings)
        print(f"service type: {type(self.service)}")
        print(f"dir(service): {dir(self.service)}")


#multikol_service_singleton = MultikolServiceSingleton(local_settings)
multikol_service_singleton = multikol_service(local_settings)

router = APIRouter()


@router.post('/spoof_test')
def variety_sample_endpoint(audio_file: UploadFile = File) -> ResponseModel:
    service = multikol_service(local_settings)
    contents = audio_file.file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp_filename = tmp.name
    tmp.close()
    #y, sr = convert_to_16k_mono_librosa(tmp_filename)
    y, sr = librosa.load(tmp_filename, sr=16000)
    os.unlink(tmp_filename)  # Delete the temporary file
    return multikol_service_singleton.inference(y)


@router.post('/spoof_test_demo')
def demo_file(audio_file: UploadFile = File) -> ResponseModel:
    contents = audio_file.file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp_filename = tmp.name
    tmp.close()
    y, sr = convert_to_16k_mono_librosa(tmp_filename)
    os.unlink(tmp_filename)  # Delete the temporary file

    return demo(y)
