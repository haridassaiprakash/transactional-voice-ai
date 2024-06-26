import argparse
import os
import time
import librosa
import soundfile as sf
import conformer
import hotword_utils
import pandas as pd
import requests
import tqdm

AUDIO_SAVE_PATH = "asr/testing/audio/"


def download_audio(URL):
    audio_name = URL.rsplit("/")[-1][:-4]
    save_path = os.path.join(AUDIO_SAVE_PATH, "{}.wav".format(audio_name))

    audio_content = requests.get(URL).content
    temp_file_path = os.path.join(AUDIO_SAVE_PATH, "temp.wav")
    with open(temp_file_path, "wb") as f:
        f.write(audio_content)
    
    y, sr = librosa.load(temp_file_path, sr=None)  
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sf.write(save_path, y, 16000)
    else:
        sf.write(save_path, y, sr) 
    
    os.remove(temp_file_path)
    y, sr = librosa.load(save_path, sr=None)
    
    return save_path


def main(args):
    hotword_mode = args.hotword_mode
    if hotword_mode == "none":
        use_hotwords = False
        hotwords = list()
    else:
        use_hotwords = True
        hotwords = hotword_utils.hotword_to_fn[hotword_mode](lang=args.lang)
    asr = conformer.ConformerRecognizer(
        model_path=args.model_path,
        lang=args.lang,
        lm_path=args.lm_path,
        alpha=args.alpha,
        beta=args.beta,
        use_hotwords=use_hotwords,
    )

    df = pd.read_csv(args.gt_file)

    urls = df["URL"]
    transcripts = list()
    transcripts_itn = list()
    transcription_time = list()
    for url in tqdm.tqdm(urls):
        fname = download_audio(url)
        if not fname:
            transcripts.append("")
            transcripts_itn.append("")
            continue
        start = time.time()
        transcript, transcript_itn = asr.transcribe(
            files=[fname],
            inference_hotwords=hotwords,
            hotword_weight=args.hotword_weight,
        )
        end = time.time()
        transcripts.append(transcript)
        transcripts_itn.append(transcript_itn)
        transcription_time.append(end - start)
    df["Transcript"] = transcripts
    df["Transcript ITN"] = transcripts_itn
    df["Transcription Time"] = transcription_time
    df.to_csv(args.save_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lang", choices=["en", "hi", "or", "ta","kn","pa","gu","bn","ml","mr","te"], required=True)
    parser.add_argument(
        "--hotword-mode",
        choices=["none"] + list(hotword_utils.hotword_to_fn.keys()),
        default="none",
    )
    parser.add_argument("--hotword-weight", type=float, default=10.0)
    parser.add_argument("--lm-path", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--save-file", required=True)
    args = parser.parse_args()
    main(args)
