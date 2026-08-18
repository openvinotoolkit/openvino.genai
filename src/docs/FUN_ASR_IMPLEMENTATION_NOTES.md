# Fun-ASR Implementation Notes

- **Original FunASR:** an unset language uses the neutral `语音转写：` prompt; a supplied language uses `语音转写成{language}：`.
- **Optimum Intel difference:** its preprocessing defaults an unset language to Chinese and uses `语音转写成中文：`. OpenVINO GenAI follows the original implementation and does not force a language by default.
- **Language result contract:** neutral FunASR transcription does not identify the spoken language. When no language is supplied, `ASRDecodedResults.languages` contains an empty string; when supplied, it contains the requested language. Automatic language detection remains model-dependent.
