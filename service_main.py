import logging
import json
from  Multikol_inference import *
import librosa

# settings default
local_settings = {"model_path": "/app/SSL_Anti-spoofing/Best_LA_model_for_DF.pth", 
                  "log_level": logging.WARNING,
                  "audio_file1": "/app/SSL_Anti-spoofing/ASVspoof2021_DF_eval/flac/DF_E_2008899_spoof.flac",
                  "audio_file2": "/app/SSL_Anti-spoofing/ASVspoof2021_DF_eval/flac/DF_E_2014017_original.flac",
                  "audio_file3": "/app/SSL_Anti-spoofing/ASVspoof2021_DF_eval/flac/shortTestRecording.wav",
                  "threshold": 0.0}

'''
DF_E_2008899 -3.683689832687378 spoof
DF_E_2014017 4.456406116485596 bonafide
'''

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    logger.info(f"logger level is {logger.getEffectiveLevel()}")

    # load model
    service = multikol_service(local_settings)

    # first file
    logger.info(f"spoof file {local_settings['audio_file1']}")
    audio, sr = librosa.load(local_settings["audio_file1"], sr=16000)

    result = service.inference(audio)
    logger.info(f"Final test result: {result}")

    # second file
    logger.info(f"bonafide file {local_settings['audio_file2']}")
    audio, sr = librosa.load(local_settings["audio_file2"], sr=16000)

    result = service.inference(audio)
    logger.info(f"Final test result: {result}")

    # third file
    logger.info(f"bonafide file {local_settings['audio_file3']}")
    audio, sr = librosa.load(local_settings["audio_file3"], sr=16000)

    result = service.inference(audio)
    logger.info(f"Final test result: {result}")
