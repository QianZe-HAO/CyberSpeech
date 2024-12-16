import ChatTTS
import torch
import torchaudio
import os

# create a folder called 'tts_output' as TTS output
if not os.path.exists('./res/tts_output'):
    os.makedirs('./res/tts_output')


chat = ChatTTS.Chat()
chat.load(compile=False)  # Set to True for better performance

texts = ["你好，俺是熊二，俺要吃蜂蜜。",
         "你好，俺是熊大，很高兴认识你!"]
wavs = chat.infer(texts)

for i in range(len(wavs)):
    """
    In some versions of torchaudio, the first line works but in other versions, so does the second line.
    """
    try:
        torchaudio.save(f"./res/tts_output/basic_output{i}.wav", torch.from_numpy(
            wavs[i]).unsqueeze(0), 24000)
    except:
        torchaudio.save(f"./res/tts_output/basic_output{i}.wav",
                        torch.from_numpy(wavs[i]), 24000)
