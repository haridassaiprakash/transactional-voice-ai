import argparse
import base64
import os
from datetime import datetime
import requests
from azure.storage.blob import BlobServiceClient, BlobClient
from flask import Flask, jsonify, request
from pipeline import PredictionPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--port", default=7861, type=int)
args = parser.parse_args()

app = Flask(__name__)

LOGGER_DB_PATH = "db/new-npci-logger.tsv"
FEEDBACK_DB_PATH = "db/npci-feedback.tsv"

account_url = "your_storage_account_url"
container_name = "your_container_name"
credential = "your_blob_credential"
prediction_pipeline = PredictionPipeline()


def upload_audio_to_blob(file_path, uuid):
    blob_name = f"{uuid}.wav"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)



def get_predictions(language, audio_data, hotword_list, hotword_weight):
    client_ip = request.remote_addr
    uuid = shortuuid.uuid()
    tmp_dir = "/tmp" 
    new_file = os.path.join(tmp_dir, f"{uuid}.wav")
    with open(new_file, "wb") as f:
        f.write(audio_data)
    print(f"File created at: {new_file}")

    predictions = prediction_pipeline.predict(
        new_file, language, hotword_list, hotword_weight
    )
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

    upload_audio_to_blob(new_file, uuid)

    return (
        predictions["transcript_itn"],
        predictions["intent"],
        predictions["entities"],
        uuid,
    )


@app.route("/api/", methods=["POST"])
def transcribe():
    input_json = request.get_json()
    language = input_json["config"]["language"]["sourceLanguage"]
    audio_uri = input_json["audio"][0].get("audioUri")
    if audio_uri:
        audio_data = requests.get(audio_uri).content
    else:
        audio_data = input_json["audio"][0].get("audioContent")
        audio_data = base64.b64decode(audio_data)

    if not input_json["config"].get("hotwords"):
        hotword_list = list()
        hotword_weight = None
    else:
        hotword_list = input_json["config"]["hotwords"].get("words", [])
        hotword_weight = input_json["config"]["hotwords"].get("weight", None)
    classlm_response = get_predictions(
        language, audio_data, hotword_list, hotword_weight
    )

    if "tag_entities" in input_json["config"]["postProcessors"]:
        ner = True
    else:
        ner = False

    if not ner:
        output_dict = {"status": "SUCCESS", "output": [{"source": classlm_response[0]}]}
        return jsonify(output_dict)
    else:
        transcript = classlm_response[0]
        intent = classlm_response[1]
        entities = classlm_response[2]
        uuid = classlm_response[3]
        output_dict = {
            "status": "SUCCESS",
            "output": [
                {
                    "source": transcript,
                    "entities": entities,
                    "intent": intent,
                    "id": uuid,
                }
            ],
        }
        return jsonify(output_dict)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)