import string
from jiwer import wer, cer
import torch
from denoiser import pretrained
import torchaudio

from denoiser.demucs import Demucs
import os
import hydra
import glob
from pesq import pesq
from pystoi.stoi import stoi
import json
import soundfile as sf
import numpy as np
from python_speech_features import mfcc

if torch.cuda.is_available():
    torchaudio.set_audio_backend("soundfile")

def _run_metrics(clean, estimate, sr):
    estimate = estimate.numpy()
    clean = clean.numpy()
    pesq_i = get_pesq(clean, estimate, sr=sr)
    stoi_i = get_stoi(clean, estimate, sr=sr)
    mcd_i = get_mcd(clean, estimate, sr=sr)
    snr_i = get_snr(clean, estimate)
    psnr_i = get_psnr(clean, estimate)

    return pesq_i, stoi_i, mcd_i, snr_i, psnr_i


def get_pesq(ref_sig, out_sig, sr):
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val


def get_mcd(ref_sig, out_sig, sr):
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += calculate_mcd(ref_sig[i], out_sig[i], sr)
    return stoi_val


def calculate_mcd(ref, deg, fs):
    ref_mfcc = mfcc(ref, samplerate=fs, numcep=13)
    deg_mfcc = mfcc(deg, samplerate=fs, numcep=13)

    min_len = min(len(ref_mfcc), len(deg_mfcc))
    ref_mfcc = ref_mfcc[:min_len]
    deg_mfcc = deg_mfcc[:min_len]

    diff = ref_mfcc - deg_mfcc
    dist = np.sqrt((diff ** 2).sum(axis=1))
    mcd = (10.0 / np.log(10)) * np.mean(dist)
    return mcd


def get_snr(ref_sig, out_sig):
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += calculate_snr(ref_sig[i], out_sig[i])
    return stoi_val


def calculate_snr(ref, deg):
    noise = ref - deg
    snr = 10 * np.log10(np.sum(ref ** 2) / np.sum(noise ** 2))
    return snr


def get_psnr(ref_sig, out_sig):
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += calculate_psnr(ref_sig[i], out_sig[i])
    return stoi_val


def calculate_psnr(ref, deg):
    mse = np.mean((ref - deg) ** 2)
    max_val = np.max(ref)
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

def extract_substring(text, s1, s2):
    # 查找s1和s2的位置
    start = text.find(s1)
    end = text.find(s2, start + len(s1))

    if start != -1 and end != -1:
        return text[start + len(s1): end]
    else:
        return ""  # 如果没有找到s1或s2，返回空字符串


def process_string(text):
    # 去掉所有标点符号
    text_no_punctuation = ''.join([char for char in text if char not in string.punctuation])
    # 转换为大写
    return text_no_punctuation.upper()

@hydra.main(config_path="conf/config.yaml")
def evaluateOneSample(args):
    print("当前工作目录：", os.getcwd())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = 'best.th'
    assert os.path.exists(weight_path), f"权重文件不存在: {weight_path}"
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = state_dict["state"]

    model_cus = Demucs(**args.demucs, sample_rate=args.sample_rate)
    model_cus.load_state_dict(state_dict)
    model_cus = model_cus.to(device)
    model_cus.eval()

    wav, sr = torchaudio.load('../../rx_signal_bin.wav')
    wav = wav.unsqueeze(0)
    wav = wav.to(device)
    with torch.no_grad():
        enhanced_cus = model_cus(wav)[0]

    if isinstance(enhanced_cus, torch.Tensor):
        enhanced_cus = enhanced_cus.detach().cpu().numpy().squeeze()

    sf.write("enhanced_cus.wav", enhanced_cus, samplerate=16000)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    evaluateOneSample()
