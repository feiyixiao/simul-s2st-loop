# Minimal simultaneous speech-to-speech translation loop (EN speech → DE speech)

A small cascade that translates while the speaker is still talking, plus the evaluation around it.

```
English audio stream ──► Whisper (base.en), re-decoded every 1 s
                          └─ LocalAgreement-2: a word is final once two hypotheses agree
                        ──► opus-mt-en-de, wait-k on committed source words
                          └─ target words committed when two translations agree, ≤ n_src − k + 1
                        ──► macOS `say` (German voice) speaks each new target chunk
```

Evaluation per policy: BLEU and chrF of the committed text, **ASR-BLEU** (Whisper-small transcribes
the synthesised German speech, then BLEU against the reference), and **Average Lagging** (AL, seconds of
source audio) of target words. `offline` waits for the full utterance.

## Result (40 WMT14 en-de sentences, 6–20 words, 1 s chunks)

| policy  | BLEU | chrF | ASR-BLEU | AL (s) | wall time / sentence (s) |
|---------|-----:|-----:|---------:|-------:|-------------------------:|
| offline | 16.3 | 47.2 | 14.7 | 4.65 | 2.1 |
| wait-1  | 16.5 | 46.4 | 14.2 | 2.97 | 3.5 |
| wait-3  | 16.5 | 46.6 | 14.2 | 3.04 | 3.4 |
| wait-5  | 16.8 | 46.9 | 14.7 | 3.38 | 3.4 |

- Simultaneous decoding cuts lagging by about 1.3–1.7 s (≈ 30–36 %) with BLEU unchanged and chrF up to 0.8 lower.
- k matters little here because LocalAgreement already holds back unstable words.
- Turning text back into speech and re-recognising it costs about 0.5–1.1 ASR-BLEU points.

## Limitations (read before quoting the numbers)
- 40 sentences, one run; differences of < 1 BLEU are within noise.
- Source speech is synthesised by macOS `say` (clean read speech), not real conversation.
- AL is measured in source-audio time and ignores computation time; re-decoding every second makes
  the streaming policies slower in wall-clock time on this laptop.
- Absolute BLEU is low: news sentences with names, a small ASR model and a small MT model.

## Run
```bash
pip install openai-whisper transformers sentencepiece sacrebleu soundfile
python -m sacrebleu -t wmt14/full -l en-de --echo src > data/wmt14.en
python -m sacrebleu -t wmt14/full -l en-de --echo ref > data/wmt14.de
python simul_s2st.py --n 40 --chunk 1.0 --ks 1,3,5
```
Results: `out/results_*.json` (all hypotheses, ASR transcripts and per-sentence AL).
