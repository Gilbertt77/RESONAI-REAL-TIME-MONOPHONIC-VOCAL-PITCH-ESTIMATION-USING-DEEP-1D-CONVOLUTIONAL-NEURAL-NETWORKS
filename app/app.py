import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import os
from collections import deque, Counter #untuk Buffer
#from google.colab import drive

#MOUNT & SETUP
#if not os.path.exists('/content/drive'):
    #drive.mount('/content/drive')

device = torch.device("cpu")

#MODEL
class VocalTunaNet(nn.Module):
    def __init__(self):
        super(VocalTunaNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 128, 51, 4, 25), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(128, 128, 51, 1, 25), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 51, 1, 25), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, 51, 1, 25), nn.BatchNorm1d(512), nn.ReLU(), nn.MaxPool1d(2))
        self.fc = nn.Linear(512 * 16, 360)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = VocalTunaNet().to(device)
state_dict = torch.load('vocal_tuna_best.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def bin_to_note_streaming(bin_idx):
    if bin_idx == 0: return "Silence", 0
    cents = bin_idx * 20
    freq = 10 * (2 ** (cents / 1200.0))
    midi = 69 + 12 * np.log2(freq / 440.0)
    note_idx = int(np.round(midi)) % 12
    note_name = NOTES[note_idx]

    exact_midi = midi
    nearest_midi = round(midi)
    cents_error = (exact_midi - nearest_midi) * 100
    return note_name, cents_error

prediction_buffer = deque(maxlen=8)

def live_tuner_stable(audio_chunk):
    if audio_chunk is None: return "..."

    sr, y = audio_chunk
    if len(y.shape) > 1: y = np.mean(y, axis=1) # Stereo to Mono

    #Normalisasi & Padding
    max_val = np.max(np.abs(y))
    if max_val > 0: y = y / max_val
    else: return "🔇 Hening"

    target_len = 1024
    if len(y) < target_len: y = np.pad(y, (0, target_len - len(y)))
    else: y = y[-target_len:]

    #Prediksi
    tensor_input = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor_input)
        current_bin = torch.argmax(output, dim=1).item()

    #Masukkan hasil sekarang ke antrian buffer
    prediction_buffer.append(current_bin)

    #Lakukan Voting (Cari bin yang paling sering muncul di buffer)
    #Ini mencegah error sesaat (glitch)
    if len(prediction_buffer) > 0:
        stable_bin, _ = Counter(prediction_buffer).most_common(1)[0]
    else:
        stable_bin = current_bin

    #Konversi Bin Stabil ke Nama Nada
    note_name, error = bin_to_note_streaming(stable_bin)

    if note_name == "Silence": return "..."

    if abs(error) < 15: indicator = "✅ TEPAT!"
    elif error > 0:     indicator = f"⬇️ Ketinggian"
    else:               indicator = f"⬆️ Kerendahan"

    return f"🎵 {note_name}\n{indicator}"

#RUN
demo = gr.Interface(
    fn=live_tuner_stable,
    inputs=gr.Audio(sources=["microphone"], type="numpy", streaming=True),
    outputs=gr.Label(label="Stabilized Output", num_top_classes=1),
    live=True,
    title="ResonAI Live Tuner",
    description="Bernyanyilah! Indikatornya akan berubah secara real-time (Latency bergantung pada koneksi internet)."
)
if __name__ == "__main__":
    demo.launch()
