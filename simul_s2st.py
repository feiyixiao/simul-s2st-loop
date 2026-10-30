"""
Minimal simultaneous speech-to-speech translation loop (English speech -> German speech).

Cascade, run incrementally on a simulated audio stream:
  1. streaming ASR   Whisper re-decodes the audio received so far every CHUNK seconds;
                     words are committed with LocalAgreement-2 (a word is final once two
                     consecutive hypotheses agree on it).
  2. simultaneous MT wait-k on committed source words: the committed source prefix is
                     re-translated (opus-mt-en-de) and target words are committed when two
                     consecutive translations agree, never more than (n_src - k + 1) of them.
                     At the end of the utterance the full translation is committed.
  3. TTS             every newly committed target chunk is spoken with macOS `say` (German voice).
Evaluation (per policy): BLEU / chrF of the committed text, ASR-BLEU (Whisper transcribes the
synthesised German speech), and Average Lagging (AL, seconds) of target words against the
source audio, plus an offline baseline that waits for the full utterance.

Source speech is synthesised from WMT14 en-de test sentences with a macOS English voice, so the
input is clean read speech (a limitation, not a claim about real conversation).

Usage:
    python simul_s2st.py --n 40 --chunk 1.0 --ks 1,3,5
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import sacrebleu
import soundfile as sf
import torch
import whisper
from transformers import MarianMTModel, MarianTokenizer

SR = 16000
HERE = Path(__file__).parent


def tts(text, voice, path):
    aiff = Path(path).with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", str(SR), "-ac", "1", str(path)], check=True)
    aiff.unlink()
    y, _ = sf.read(path, dtype="float32")
    return y


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def norm(w):
    return w.strip(".,!?;:\"'").lower()


class Pipeline:
    def __init__(self, asr_name, device):
        self.asr = whisper.load_model(asr_name, device="cpu")
        self.tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.mt = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device).eval()
        self.device = device

    def transcribe(self, audio, lang="en"):
        r = self.asr.transcribe(audio, language=lang, fp16=False, temperature=0.0, condition_on_previous_text=False)
        return r["text"].strip()

    @torch.no_grad()
    def translate(self, text):
        if not text:
            return ""
        b = self.tok([text], return_tensors="pt").to(self.device)
        out = self.mt.generate(**b, num_beams=4, max_new_tokens=128)
        return self.tok.decode(out[0], skip_special_tokens=True)


def run_stream(pipe, audio, chunk, k):
    """Return committed target words with emission times (seconds of source audio received)."""
    dur = len(audio) / SR
    src_commit, prev_hyp = [], []
    tgt_commit, tgt_times, prev_tr = [], [], []
    t = chunk
    while True:
        t = min(t, dur)
        final = t >= dur
        hyp = pipe.transcribe(audio[: int(t * SR)]).split()
        if final:
            src_commit = hyp
        else:
            n = common_prefix([norm(w) for w in prev_hyp], [norm(w) for w in hyp])
            if n > len(src_commit):
                src_commit = hyp[:n]
        prev_hyp = hyp
        if final:
            tr = pipe.translate(" ".join(src_commit)).split()
            new = tr[len(tgt_commit):] if tr[: len(tgt_commit)] == tgt_commit else tr[len(tgt_commit):]
            tgt_commit += new
            tgt_times += [dur] * len(new)
            break
        if len(src_commit) >= k:
            tr = pipe.translate(" ".join(src_commit)).split()
            agree = common_prefix(prev_tr, tr)
            limit = len(src_commit) - k + 1
            n_new = min(agree, limit) - len(tgt_commit)
            if n_new > 0 and tr[: len(tgt_commit)] == tgt_commit:
                tgt_commit += tr[len(tgt_commit): len(tgt_commit) + n_new]
                tgt_times += [t] * n_new
            prev_tr = tr
        t += chunk
    return tgt_commit, tgt_times, dur


def average_lagging(times, src_dur, n_ref):
    """AL in seconds (Ma et al. 2019), delays measured in source-audio seconds."""
    if not times:
        return float("nan")
    rate = src_dur / max(n_ref, 1)
    tau = next((i for i, d in enumerate(times, 1) if d >= src_dur), len(times))
    return float(np.mean([times[i - 1] - (i - 1) * rate for i in range(1, tau + 1)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max_words", type=int, default=20)
    ap.add_argument("--chunk", type=float, default=1.0)
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--asr", default="base.en")
    ap.add_argument("--src_voice", default="Samantha")
    ap.add_argument("--tgt_voice", default="Anna")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    en = (HERE / "data/wmt14.en").read_text().splitlines()
    de = (HERE / "data/wmt14.de").read_text().splitlines()
    pairs = [(s, r) for s, r in zip(en, de) if 6 <= len(s.split()) <= args.max_words][: args.n]
    print(f"{len(pairs)} sentence pairs, chunk {args.chunk}s, ASR whisper-{args.asr}, MT opus-mt-en-de on {device}")

    pipe = Pipeline(args.asr, device)
    asr_de = whisper.load_model("small", device="cpu")
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())

    policies = ["offline"] + [f"wait-{k}" for k in args.ks.split(",")]
    res = {p: {"hyp": [], "asr": [], "al": [], "t": []} for p in policies}
    refs = []
    for i, (src, ref) in enumerate(pairs):
        audio = tts(src, args.src_voice, tmp / f"src{i}.wav")
        refs.append(ref)
        for p in policies:
            t0 = time.time()
            if p == "offline":
                words = pipe.translate(pipe.transcribe(audio)).split()
                times, dur = [len(audio) / SR] * len(words), len(audio) / SR
            else:
                words, times, dur = run_stream(pipe, audio, args.chunk, int(p.split("-")[1]))
            hyp = " ".join(words)
            speech = tts(hyp, args.tgt_voice, tmp / f"tgt_{p}_{i}.wav") if hyp else np.zeros(SR, dtype="float32")
            back = asr_de.transcribe(speech, language="de", fp16=False, temperature=0.0)["text"].strip()
            res[p]["hyp"].append(hyp)
            res[p]["asr"].append(back)
            res[p]["al"].append(average_lagging(times, dur, len(ref.split())))
            res[p]["t"].append(time.time() - t0)
        if i < 3 or (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(pairs)}] {src}\n   ref     : {ref}\n   offline : {res['offline']['hyp'][-1]}\n   {policies[-1]:8s}: {res[policies[-1]]['hyp'][-1]}", flush=True)

    summary = {}
    for p in policies:
        summary[p] = {
            "BLEU": round(sacrebleu.corpus_bleu(res[p]["hyp"], [refs]).score, 2),
            "chrF": round(sacrebleu.corpus_chrf(res[p]["hyp"], [refs]).score, 2),
            "ASR-BLEU": round(sacrebleu.corpus_bleu(res[p]["asr"], [refs]).score, 2),
            "AL_s": round(float(np.nanmean(res[p]["al"])), 3),
            "wall_s_per_sent": round(float(np.mean(res[p]["t"])), 2),
        }
    print("\npolicy     BLEU   chrF  ASR-BLEU   AL(s)  wall(s)")
    for p, m in summary.items():
        print(f"{p:9s} {m['BLEU']:6.2f} {m['chrF']:6.2f} {m['ASR-BLEU']:8.2f} {m['AL_s']:7.2f} {m['wall_s_per_sent']:7.2f}")
    stamp = time.strftime("%Y%m%dT%H%M")
    (out_dir / f"results_{stamp}.json").write_text(json.dumps(
        {"args": vars(args), "summary": summary, "refs": refs, "sources": [s for s, _ in pairs],
         "outputs": {p: {"hyp": res[p]["hyp"], "asr": res[p]["asr"], "al": res[p]["al"]} for p in policies}},
        indent=1, ensure_ascii=False))
    print("saved →", out_dir / f"results_{stamp}.json")


if __name__ == "__main__":
    main()
