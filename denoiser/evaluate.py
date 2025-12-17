# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys
import math
from pesq import pesq
from pystoi import stoi
import torch

from .data import NoisyCleanSet
from .enhance import add_flags, get_estimate
from . import distrib, pretrained
from .utils import bold, LogProgress
from python_speech_features import mfcc
import numpy as np
from demucs import Demucs

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    'denoiser.evaluate',
    description='Speech enhancement using Demucs - Evaluate model performance')
add_flags(parser)
parser.add_argument('--data_dir', help='directory including noisy.json and clean.json files')
parser.add_argument('--matching', default="sort", help='set this to dns for the dns dataset.')
parser.add_argument('--no_pesq', action="store_false", dest="pesq", default=True,
                    help="Don't compute PESQ.")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")


def evaluate(args, model=None, data_loader=None):
    total_pesq = 0
    total_stoi = 0
    total_mcd = 0
    total_snr = 0
    total_psnr = 0
    total_cnt = 0
    updates = 5

    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()

    # Load data
    if data_loader is None:
        dataset = NoisyCleanSet(args.data_dir,
                                matching=args.matching, sample_rate=model.sample_rate)
        data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)
    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                noisy, clean = [x.to(args.device) for x in data]
                # If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(_estimate_and_run_metrics, clean, model, noisy, args))
                else:
                    estimate = get_estimate(model, noisy, args)
                    # estimate = noisy
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(_run_metrics, clean, estimate, args, model.sample_rate))
                total_cnt += clean.shape[0]

        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            try:
                pesq_i, stoi_i, mcd_i, snr_i, psnr_i = pending.result()

                # 如果结果本身为 NaN，也跳过
                if math.isnan(pesq_i) or math.isnan(stoi_i):
                    logger.warning("NaN detected in PESQ/STOI result, skipping.")
                    continue

                total_pesq += pesq_i
                total_stoi += stoi_i
                total_mcd += mcd_i
                total_snr += snr_i
                total_psnr += psnr_i

            except ValueError as e:
                logger.warning(f"Error during PESQ/STOI computation: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error in evaluation thread: {e}")
                continue

    metrics = [total_pesq, total_stoi, total_mcd, total_snr, total_psnr]
    pesq, stoi, mcd, snr, psnr = distrib.average([m / total_cnt for m in metrics], total_cnt)
    logger.info(bold(f'Test set performance:PESQ={pesq}, STOI={stoi}, MCD={mcd}, SNR={snr}, PSNR={psnr}.'))
    return pesq, stoi, mcd, snr, psnr


def _estimate_and_run_metrics(clean, model, noisy, args):
    estimate = get_estimate(model, noisy, args)
    return _run_metrics(clean, estimate, args, sr=model.sample_rate)


def _run_metrics(clean, estimate, args, sr):
    estimate = estimate.numpy()[:, 0]
    clean = clean.numpy()[:, 0]
    if args.pesq:
        pesq_i = get_pesq(clean, estimate, sr=sr)
    else:
        pesq_i = 0
    stoi_i = get_stoi(clean, estimate, sr=sr)
    mcd_i = get_mcd(clean, estimate, sr=sr)
    snr_i = get_snr(clean, estimate)
    psnr_i = get_psnr(clean, estimate)

    return pesq_i, stoi_i, mcd_i, snr_i, psnr_i


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    model = Demucs(**args.demucs, sample_rate=args.sample_rate)
    weight_path = 'best.th'
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = state_dict["state"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    kwargs = {"matching": args.dset.matching, "sample_rate": args.sample_rate}
    tt_dataset = NoisyCleanSet(args.dset.test, **kwargs)
    tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)

    pesq, stoi, mcd, snr, psnr = evaluate(args, model, tt_loader)
    print(f'pesq: {pesq}, stoi: {stoi}, mcd: {mcd}, snr: {snr}, psnr: {psnr}')

if __name__ == '__main__':
    main()
