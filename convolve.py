# !usr/bin/env python3

import argparse
import cProfile
from math import log2
import pstats
import struct
import wave
from array import array
from itertools import batched
from wave import _wave_params
from four1 import four1


def float_to_short(f: float) -> int:
    r = int(f * 32768)
    if r >= 32768:
        r = 32767
    elif r < -32768:
        r = -32768
    return r


# fmt: off

# DFT.
def convolution(x: list[float], h: list[float]) -> list[float]:
    k = len(x)
    y = [0.0] * k
    for m, n in batched(range(k), 2):
        u, v, q, r = x[m], x[n], h[m], h[n]
        y[m], y[n] = u*q - v*r, u*r + v*q
    return y

def convolve(xs: list[float], hs: list[float]) -> list[float]:
    M, N = len(xs), len(hs)
    K = 1 << int(log2(M*2 - 1) + 1)
    P = M+N - 1

    x = [0.0] * K*2
    h = [0.0] * K*2

    x[0:M*2:2] = xs
    h[0:N*2:2] = hs

    x = four1(x, K, 1)
    h = four1(h, K, 1)
    y = four1(convolution(x, h), K, -1)
    
    return [y[i] / K for i in range(P*2)[::2]]
# fmt: on


def read_wav(path: str) -> tuple[_wave_params, list[float]]:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("File must be mono channel.")
        size = wf.getnframes()
        raw = wf.readframes(size)
        params = wf.getparams()
    return params, [x / 32768 for x, in struct.iter_unpack("<h", raw)]


def write_wav(path: str, params: _wave_params, data: list[float]) -> None:
    with wave.open(path, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(array("h", map(float_to_short, data)).tobytes())


def main(inputfile: str, irfile: str, outputfile: str) -> None:
    input_params, input_data = read_wav(inputfile)
    impulse_data = read_wav(irfile)[1]
    output_data = convolve(input_data, impulse_data)

    lim = max(1, max(map(abs, output_data)))
    output_data[:] = (x / lim for x in output_data)  # Normalize values.

    write_wav(outputfile, input_params, output_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolve two WAV files.")
    parser.add_argument(
        "inputfile",
        help="The path of the input file.",
        type=str,
    )
    parser.add_argument(
        "irfile",
        help="The path of the impulse response file.",
        type=str,
    )
    parser.add_argument(
        "outputfile",
        help="The path of the output file.",
        type=str,
    )
    args = parser.parse_args()
    with cProfile.Profile() as pr:
        main(args.inputfile, args.irfile, args.outputfile)
    pr.print_stats(pstats.SortKey.TIME)
