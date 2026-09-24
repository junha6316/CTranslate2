import numpy as np
from faster_whisper import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor

def windows(n_mels):
    fe = FeatureExtractor(feature_size=n_mels)
    out = []
    for p in ["/opt/audio/physicsworks.wav", "/opt/audio/jfk_x5.flac"]:
        f = fe(decode_audio(p, sampling_rate=16000))
        for s in range(0, f.shape[-1], 3000):
            w = f[..., s:s + 3000]
            if w.shape[-1] < 1000:   # drop tails shorter than 10 s
                continue
            if w.shape[-1] < 3000:
                w = np.pad(w, ((0, 0), (0, 3000 - w.shape[-1])))
            out.append(w.astype(np.float32))
    return out

