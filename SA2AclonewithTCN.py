#!/usr/bin/env python3
# sa2clone_gui_tcn_inference.py
# Requirements: torch, torchaudio (optional), librosa, soundfile, numpy, PyQt5, matplotlib
# Usage: python sa2clone_gui_tcn_inference.py
# Select your model file (e.g. tcn_direct_best.pth) in the GUI.

import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QSlider, QLabel, QFileDialog, QTabWidget,
                             QMessageBox, QFrame)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------
# Recreate exact training model classes
# ---------------------------

class PeakCond(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1,32), nn.SiLU(), nn.Linear(32,out_dim), nn.SiLU())
    def forward(self, peak, T):
        # peak : shape (B,1) or (B,) or (1,) etc.
        if peak.dim() == 1:
            peak = peak.unsqueeze(-1)   # (B,1)
        p = torch.clamp(peak / 100.0, 0.0, 1.0)
        feat = self.mlp(p).unsqueeze(-1)   # (B, out_dim, 1)
        return feat.expand(-1, -1, T)      # (B, out_dim, T)

class DilatedBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        pad = dilation
        self.conv_f = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.conv_g = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.res = nn.Conv1d(ch, ch, 1)
        self.skip = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        h = f * g
        return self.res(h) + x, self.skip(h)

class TCNModel(nn.Module):
    def __init__(self, res_ch=64, cond_dim=32, layers=8, stacks=2):
        super().__init__()
        self.inp = nn.Conv1d(1, res_ch, 1)
        self.cond = PeakCond(cond_dim)
        self.cond_proj = nn.Conv1d(cond_dim, res_ch, 1)
        self.blocks = nn.ModuleList([DilatedBlock(res_ch, 2 ** (i % layers)) for i in range(layers * stacks)])
        self.out = nn.Sequential(nn.ReLU(), nn.Conv1d(res_ch, res_ch, 1), nn.ReLU(), nn.Conv1d(res_ch, 1, 1))

    def forward(self, x, peak):
        # x: [B, C, T] where C==1
        B, C, T = x.shape
        h = self.inp(x)                       # [B, res_ch, T]
        c = self.cond(peak, T)                # [B, cond_dim, T]
        h = h + self.cond_proj(c)             # project cond to res_ch and add
        skips = []
        for blk in self.blocks:
            h, s = blk(h)
            skips.append(s)
        h = sum(skips) / math.sqrt(len(skips))
        y = self.out(h)
        return y  # [B,1,T]

# ---------------------------
# Helper: robust checkpoint loader (handles common saved dict formats)
# ---------------------------
def load_checkpoint_to_model(path, model, map_location='cpu'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ck = torch.load(path, map_location=map_location)
    # Cases:
    # - ck is state_dict
    # - ck is dict with 'model_state'/'model_state_dict'/'state_dict'/'model'
    # - ck is full nn.Module (rare)
    if isinstance(ck, nn.Module):
        return ck
    if isinstance(ck, dict):
        # try keys
        for k in ("model_state","model_state_dict","state_dict","model"):
            if k in ck:
                try:
                    model.load_state_dict(ck[k])
                    return model
                except Exception:
                    pass
        # fallback: ck might itself be a raw state_dict
        sample_keys = list(ck.keys())[:5]
        if len(sample_keys) > 0 and isinstance(ck[sample_keys[0]], torch.Tensor):
            model.load_state_dict(ck)
            return model
    # last attempt: try loading directly (may raise)
    model.load_state_dict(ck)
    return model

# ---------------------------
# Processing class — inference that matches training exactly (no overlap)
# ---------------------------
class SA2CloneInference:
    def __init__(self, model_path=None, device=None,
                 clip_len=512, hop_len=512, sr_train=44100):
        self.device = device if device is not None else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
        self.clip_len = int(clip_len)
        self.hop_len = int(hop_len)
        self.sr_train = sr_train

        # create model matching training hyperparams
        self.model = TCNModel(res_ch=64, cond_dim=32, layers=8, stacks=2).to(self.device)
        self.loaded = False
        if model_path:
            self.load_model(model_path)
        self.model.eval()

        # compute maximum dilation for padding strategy
        self.max_dilation = 2 ** (8 - 1)  # layers=8 -> largest dilation 2^(layers-1)
        # safe pad both sides
        self.edge_pad = self.max_dilation * 2

    def load_model(self, model_path):
        try:
            load_checkpoint_to_model(model_path, self.model, map_location=self.device)
            self.loaded = True
            print(f"[SA2CloneInference] loaded model from {model_path} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def process_file(self, in_wav, peak_value, out_wav, input_gain=1.0, output_gain=1.0, resample_to_train=True):
        """
        in_wav: input path
        peak_value: scalar (0..100)
        out_wav: output path
        Non-overlapping processing: step == clip_len
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded.")

        # 1) load audio (mono)
        audio, sr = librosa.load(in_wav, sr=None, mono=True)
        if resample_to_train and sr != self.sr_train:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr_train)
            sr = self.sr_train

        # apply external input gain
        audio = audio.astype(np.float32) * float(input_gain)

        N = len(audio)
        W = self.clip_len
        H = self.hop_len
        assert H == W, "This inference path expects no overlap: hop == clip"

        # pad both sides to reduce boundary artifacts (model uses dilated convs)
        pad_L = self.edge_pad
        pad_R = (W - (N % W)) % W + self.edge_pad  # ensure final frame fits exactly
        audio_p = np.concatenate([np.zeros(pad_L, dtype=np.float32), audio, np.zeros(pad_R, dtype=np.float32)])
        out_p = np.zeros_like(audio_p, dtype=np.float32)

        # prepare peak tensor shape [B,1] -> for single-frame B=1
        # We'll reuse same peak for every clip
        peak_t_single = torch.tensor([[float(peak_value)]], dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # process sequential non-overlapping frames
            for start in range(0, len(audio_p) - W + 1, H):
                frame = audio_p[start:start+W].astype(np.float32)
                x = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,W]
                # forward: model(x, peak)
                out_t = self.model(x, peak_t_single)  # expected [1,1,W]
                y = out_t.cpu().numpy().squeeze(0).squeeze(0).astype(np.float32)  # [W]
                out_p[start:start+W] = y  # no overlap-add, just sequential placement

        # crop to original length
        result = out_p[pad_L:pad_L+N]
        result = result * float(output_gain)

        # save
        sf.write(out_wav, result.astype(np.float32), sr, subtype='FLOAT')
        return audio, result, sr

# ---------------------------
# Plot widget
# ---------------------------
class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Default customizable settings
        self.font_size = 12
        self.fig_width = 6
        self.fig_height = 4

        self.figure = Figure(figsize=(self.fig_width, self.fig_height))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def set_font_size(self, fs):
        self.font_size = fs

    def set_fig_size(self, w, h):
        self.fig_width = w
        self.fig_height = h
        self.figure.set_size_inches(w, h, forward=True)

    def save_figure(self, filename):
        self.figure.savefig(filename, dpi=200, bbox_inches='tight')

    def plot_spectrogram(self, audio, sr=44100, title="Spectrogram"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        logS = librosa.amplitude_to_db(S, ref=np.max)
        extent = [0, len(audio)/sr, 0, sr/2]

        im = ax.imshow(logS, aspect='auto', origin='lower',
                       cmap='magma', extent=extent)

        ax.set_title(title, fontsize=self.font_size)
        ax.set_xlabel('Time (s)', fontsize=self.font_size)
        ax.set_ylabel('Frequency (Hz)', fontsize=self.font_size)
        self.figure.colorbar(im, ax=ax, format='%+2.0f dB')

        self.canvas.draw()

    def plot_waveform(self, audio, sr=44100, title="Waveform"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        times = np.arange(len(audio)) / float(sr)
        ax.plot(times, audio)

        ax.set_title(title, fontsize=self.font_size)
        ax.set_xlabel('Time (s)', fontsize=self.font_size)
        ax.set_ylabel('Amplitude', fontsize=self.font_size)

        self.canvas.draw()


# ---------------------------
# GUI
# ---------------------------
class SA2CloneGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SA2Clone TCN Inference (no-overlap)")
        self.setGeometry(40, 40, 1400, 900)

        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        print(f"[GUI] device: {self.device}")

        self.processor = None
        self.model_path = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        # model select
        h = QHBoxLayout()
        btn_model = QPushButton("Select model (.pth)")
        btn_model.clicked.connect(self.on_select_model)
        h.addWidget(btn_model)
        self.lbl_model = QLabel("No model selected")
        h.addWidget(self.lbl_model)
        h.addStretch()
        v.addLayout(h)

        # file selectors
        frame = QFrame()
        fl = QVBoxLayout(frame)

        ih = QHBoxLayout()
        self.btn_in = QPushButton("Select input WAV")
        self.btn_in.clicked.connect(self.on_select_input)
        ih.addWidget(self.btn_in)
        self.lbl_in = QLabel("No input")
        ih.addWidget(self.lbl_in)
        ih.addStretch()
        fl.addLayout(ih)

        oh = QHBoxLayout()
        self.btn_out = QPushButton("Select output WAV")
        self.btn_out.clicked.connect(self.on_select_output)
        oh.addWidget(self.btn_out)
        self.lbl_out = QLabel("No output")
        oh.addWidget(self.lbl_out)
        oh.addStretch()
        fl.addLayout(oh)

        v.addWidget(frame)

        # controls: input gain, output gain, peak
        ctrl = QFrame()
        cl = QHBoxLayout(ctrl)

        # =====================
        # Input Gain (dB)
        # =====================
        cl.addWidget(QLabel("Input Gain (dB):"))
        self.s_in_gain = QSlider(Qt.Horizontal)
        self.s_in_gain.setRange(-240, 240)   # -24.0 dB ~ +24.0 dB (0.1dB step)
        self.s_in_gain.setValue(0)           # default = 0dB
        self.s_in_gain.valueChanged.connect(self.update_in_gain_label)
        cl.addWidget(self.s_in_gain)
        self.lbl_in_gain = QLabel("0.0 dB")
        cl.addWidget(self.lbl_in_gain)

        # =====================
        # Output Gain (dB)
        # =====================
        cl.addWidget(QLabel("Output Gain (dB):"))
        self.s_out_gain = QSlider(Qt.Horizontal)
        self.s_out_gain.setRange(-240, 240)
        self.s_out_gain.setValue(0)
        self.s_out_gain.valueChanged.connect(self.update_out_gain_label)
        cl.addWidget(self.s_out_gain)
        self.lbl_out_gain = QLabel("0.0 dB")
        cl.addWidget(self.lbl_out_gain)

        cl.addSpacing(20)
        cl.addWidget(QLabel("Peak (0..100):"))
        self.s_peak = QSlider(Qt.Horizontal); self.s_peak.setRange(0,100); self.s_peak.setValue(50)
        self.s_peak.valueChanged.connect(lambda val: self.lbl_peak.setText(str(val)))
        cl.addWidget(self.s_peak)
        self.lbl_peak = QLabel("50"); cl.addWidget(self.lbl_peak)

        cl.addStretch()
        v.addWidget(ctrl)

        # === Visualization settings (font size & figure size) ===
        vis_frame = QFrame()
        vis_layout = QHBoxLayout(vis_frame)

        # Font size
        vis_layout.addWidget(QLabel("Font Size:"))
        self.s_font = QSlider(Qt.Horizontal)
        self.s_font.setRange(8, 24)
        self.s_font.setValue(12)
        self.s_font.valueChanged.connect(self.on_font_change)
        vis_layout.addWidget(self.s_font)

        # Figure width
        vis_layout.addWidget(QLabel("Figure Width:"))
        self.s_fig_w = QSlider(Qt.Horizontal)
        self.s_fig_w.setRange(4, 12)
        self.s_fig_w.setValue(6)
        self.s_fig_w.valueChanged.connect(self.on_fig_size_change)
        vis_layout.addWidget(self.s_fig_w)

        # Figure height
        vis_layout.addWidget(QLabel("Figure Height:"))
        self.s_fig_h = QSlider(Qt.Horizontal)
        self.s_fig_h.setRange(3, 10)
        self.s_fig_h.setValue(4)
        self.s_fig_h.valueChanged.connect(self.on_fig_size_change)
        vis_layout.addWidget(self.s_fig_h)

        vis_layout.addStretch()
        v.addWidget(vis_frame)

        # process button
        self.btn_process = QPushButton("Process (offline, no-overlap)")
        self.btn_process.clicked.connect(self.on_process)
        v.addWidget(self.btn_process)

        # tabs
        self.tabs = QTabWidget()
        v.addWidget(self.tabs)

        t1 = QWidget()
        t1l = QVBoxLayout(t1)

        # top area: save buttons
        btn_area = QHBoxLayout()
        self.btn_save_in_spec = QPushButton("Save Input Spectrogram")
        self.btn_save_out_spec = QPushButton("Save Output Spectrogram")
        self.btn_save_in_spec.clicked.connect(lambda: self.save_plot(self.plot_in_spec))
        self.btn_save_out_spec.clicked.connect(lambda: self.save_plot(self.plot_out_spec))
        btn_area.addWidget(self.btn_save_in_spec)
        btn_area.addWidget(self.btn_save_out_spec)
        btn_area.addStretch()

        t1l.addLayout(btn_area)

        # bottom area: two figures
        fig_area = QHBoxLayout()
        self.plot_in_spec = PlotWidget()
        self.plot_out_spec = PlotWidget()
        fig_area.addWidget(self.plot_in_spec)
        fig_area.addWidget(self.plot_out_spec)

        t1l.addLayout(fig_area)
        self.tabs.addTab(t1, "Spectrogram")

        t2 = QWidget()
        t2l = QHBoxLayout(t2)
        self.plot_in_wave = PlotWidget()
        self.plot_out_wave = PlotWidget()
        t2l.addWidget(self.plot_in_wave)
        t2l.addWidget(self.plot_out_wave)
        self.tabs.addTab(t2, "Waveform")

        # state
        self.input_file = None
        self.output_file = None
        self.input_audio = None
        self.output_audio = None

    def update_in_gain_label(self, value):
        db = value / 10.0   # slider integer → 0.1dB steps
        self.lbl_in_gain.setText(f"{db:.1f} dB")

    def update_out_gain_label(self, value):
        db = value / 10.0
        self.lbl_out_gain.setText(f"{db:.1f} dB")


    def save_plot(self, plot_widget):
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", os.getcwd(), "PNG Image (*.png)")
        if not path:
            return
        plot_widget.save_figure(path)
        QMessageBox.information(self, "Saved", f"Saved as: {path}")

    def on_font_change(self, value):
        self.plot_in_spec.set_font_size(value)
        self.plot_out_spec.set_font_size(value)
        if self.input_audio is not None:
            self.plot_in_spec.canvas.draw()
        if self.output_audio is not None:
            self.plot_out_spec.canvas.draw()

    def on_fig_size_change(self, value):
        w = self.s_fig_w.value()
        h = self.s_fig_h.value()
        self.plot_in_spec.set_fig_size(w, h)
        self.plot_out_spec.set_fig_size(w, h)
        if self.input_audio is not None:
            self.plot_in_spec.canvas.draw()
        if self.output_audio is not None:
            self.plot_out_spec.canvas.draw()

    # callbacks
    def on_select_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select model checkpoint", os.getcwd(), "PyTorch files (*.pth *.pt)")
        if not p:
            return
        self.model_path = p
        self.lbl_model.setText(os.path.basename(p))
        try:
            self.processor = SA2CloneInference(model_path=p, device=self.device, clip_len=512, hop_len=512, sr_train=44100)
            QMessageBox.information(self, "Model", "Model loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Model error", f"Failed to load model: {e}")
            self.processor = None

    def on_select_input(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select input WAV", os.getcwd(), "Audio files (*.wav *.flac *.aiff)")
        if not p:
            return
        self.input_file = p
        self.lbl_in.setText(os.path.basename(p))
        # plot input at training SR (resample for display)
        audio, sr = librosa.load(p, sr=44100, mono=True)
        self.input_audio = audio
        self.plot_in_spec.plot_spectrogram(audio, sr=sr, title="Input Spectrogram")
        self.plot_in_wave.plot_waveform(audio, sr=sr, title="Input Waveform")

    def on_select_output(self):
        p, _ = QFileDialog.getSaveFileName(self, "Select output WAV", os.getcwd(), "WAV files (*.wav)")
        if not p:
            return
        self.output_file = p
        self.lbl_out.setText(os.path.basename(p))

    def on_process(self):
        if self.processor is None:
            QMessageBox.warning(self, "Error", "Load model first.")
            return
        if not self.input_file or not self.output_file:
            QMessageBox.warning(self, "Error", "Select input and output files.")
            return
        peak = int(self.s_peak.value())

        # dB → linear
        in_gain_db = self.s_in_gain.value() / 10.0
        out_gain_db = self.s_out_gain.value() / 10.0

        in_gain = 10 ** (in_gain_db / 20.0)
        out_gain = 10 ** (out_gain_db / 20.0)

        try:
            audio_in, audio_out, sr = self.processor.process_file(self.input_file, peak, self.output_file, input_gain=in_gain, output_gain=out_gain, resample_to_train=True)
            self.output_audio = audio_out
            # plot output (use sr)
            self.plot_out_spec.plot_spectrogram(audio_out, sr=sr, title="Output Spectrogram")
            self.plot_out_wave.plot_waveform(audio_out, sr=sr, title="Output Waveform")
            QMessageBox.information(self, "Done", f"Saved: {self.output_file}")
        except Exception as e:
            QMessageBox.critical(self, "Processing error", str(e))

# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SA2CloneGUI()
    w.show()
    sys.exit(app.exec_())
