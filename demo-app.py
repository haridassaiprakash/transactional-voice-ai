import argparse
import os
from datetime import datetime
import soundfile as sf
import gradio as gr
import shortuuid
import base64
import shutil
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from pipeline import PredictionPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--port", default=7870, type=int)
args = parser.parse_args()

GRADIO_PAGE_TITLE = "Bhashini: Tagged Speech Recognition Demo"
GRADIO_PAGE_DESC = """
<h1 style="text-align: center; margin-bottom: 1rem">Bhashini: Transactional Voice AI</h1>
"""

LOGGER_DB_PATH = "db/npci-looger-ui.tsv"
FEEDBACK_DB_PATH = "db/npci-looger-ui.tsv"

account_url = "your_storage_account_url"
container_name = "your_container_name"
credential = "your_blob_credential"
prediction_pipeline = PredictionPipeline()

def upload_audio_to_blob(file_path, uuid):
    blob_name = f"audio-{uuid}.wav"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
 
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)


def get_predictions(language, audio_file, mic_audio_file=None):

    if mic_audio_file is not None:
        # Save the NumPy audio data to a .wav file
        sample_rate, audio_data = mic_audio_file
        uuid = shortuuid.uuid()
        tmp_dir = "/tmp" 
        temp_audio_path  = os.path.join(tmp_dir, f"temp_audio_{uuid}.wav")
        sf.write(temp_audio_path, audio_data, sample_rate)
        new_file = temp_audio_path

    elif audio_file is not None:
        uuid = shortuuid.uuid()
        file = audio_file
        temp_dir = "/tmp" 
        new_file = os.path.join(temp_dir, f"{uuid}.wav")
        shutil.copyfile(audio_file, new_file)

    else:
        return "[Error - Audio File Not Provided]"

    
    predictions = prediction_pipeline.predict(new_file, language)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    row = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            uuid,
            language,
            predictions["transcript_itn"],
            predictions["entities"],
            predictions["intent"],
            predictions["intent_orig"],
            predictions["intent_prob"],
            dt_string,
    )
    with open(LOGGER_DB_PATH, "a") as f:
        f.write(row)
        
    upload_audio_to_blob(new_file,uuid)

    if mic_audio_file is not None:
        os.remove(temp_audio_path)
    return (
        {"text": predictions["transcript_itn"], "entities": predictions["entities"]},
        predictions["transcript_itn"],
        predictions["intent"],
        predictions["entities"],
        uuid,
        )


def record_feedback(feedback, lang, transcript, entities, uuid):
    if not uuid:
        return
    if len(transcript) == 0 or len(entities) == 0:
        return
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    row = "{}\t{}\t{}\t{}\t{}\t{}\n".format(
        uuid, lang, transcript, entities, feedback, dt_string
    )
    with open(FEEDBACK_DB_PATH, "a") as f:
        f.write(row)


def record_feedback_correct(lang, transcript, entities, uuid):
    record_feedback("correct", lang, transcript, entities, uuid)


def record_feedback_incorrect(lang, transcript, entities, uuid):
    record_feedback("incorrect", lang, transcript, entities, uuid)


with gr.Blocks(title=GRADIO_PAGE_TITLE) as demo:
    with gr.Row():
        gr.HTML(value=GRADIO_PAGE_DESC)
    with gr.Row():
        with gr.Column():
            in_language = gr.Radio(
                label="Language", choices=["en", "hi","mr","ta","gu","bn","kn","ml","or","pa","te"], value="English"
            )
            in_uploaded_audio = gr.File(label="Upload Speech", type="filepath")
            in_recorded_audio = gr.Audio(label="Record Speech", type="numpy")
            btn_transcribe = gr.Button(value="Transcribe")
        with gr.Column():
            out_tagged_text = gr.HighlightedText(label="Tagged Speech Recognition Output")
            out_transcript = gr.Textbox(label="Speech Recognition Output", lines=1)
            out_intent = gr.Textbox(label="Intent Recognition Output", lines=1)
            out_entities = gr.Textbox(label="Entities JSON", lines=1, visible=True)
            audio_uuid = gr.Textbox(label="Sample UUID", lines=1)
            with gr.Row():
                btn_feedback_correct = gr.Button("Correct")
                btn_feedback_incorrect = gr.Button("Incorrect")
    btn_transcribe.click(
        fn=get_predictions,
        inputs=[in_language, in_uploaded_audio, in_recorded_audio],
        outputs=[
            out_tagged_text,
            out_transcript,
            out_intent,
            out_entities,
            audio_uuid,
        ],
        api_name="transcribe",
    )
    btn_feedback_correct.click(
        fn=record_feedback_correct,
        inputs=[in_language, out_transcript, out_entities, audio_uuid,],
        api_name="feedback_correct",
    )
    btn_feedback_incorrect.click(
        fn=record_feedback_incorrect,
        inputs=[in_language, out_transcript, out_entities, audio_uuid,],
        api_name="feedback_incorrect",
    )

demo.launch(server_name="0.0.0.0", server_port=args.port, share= True)
