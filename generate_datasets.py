from typing import List

import h5py
import numpy as np
import torch

from ml4gw import gw

seed = 101588
rng = np.random.default_rng(seed)

SAMPLE_RATE = 2048

def generate_background(
    background_file: str,
    kernel_length: float = 10.5,
    num_samples: int = 100000,
    save_file: bool = True
) -> np.array:
    with h5py.File(background_file, "r") as f:
        h1 = f["H1"][:]
        l1 = f["L1"][:]

    # The background data was generated at 2048 Hz
    kernel_size = int(kernel_length * SAMPLE_RATE) 

    max_idx = h1.shape[-1] - kernel_size - 1
    h1_idxs = rng.integers(0, max_idx, num_samples)
    l1_idxs = rng.integers(0, max_idx, num_samples)

    background_samples = np.zeros((num_samples, 2, kernel_size))
    for i, (h1_idx, l1_idx) in enumerate(zip(h1_idxs, l1_idxs)):
        background_samples[i, 0] = h1[h1_idx : h1_idx + kernel_size]
        background_samples[i, 1] = l1[l1_idx : l1_idx + kernel_size]

    if save_file:
        with h5py.File("background.hdf5", "w") as f:
            f.create_dataset("data", data=background_samples)

    return background_samples


def generate_injections(
    waveform_file: str,
    background_file: str,
) -> None:
    with h5py.File(waveform_file, "r") as f:
        signals = torch.Tensor(f["signals"][:])
        dec = torch.Tensor(f["dec"][:])
        phi = torch.Tensor(f["ra"][:] - np.pi)
        psi = torch.Tensor(f["psi"][:])
        coalescence_idx = int(f.attrs["coalescence_time"] * SAMPLE_RATE)

    polarizations = {
        "cross": signals[:, 0],
        "plus": signals[:, 1],
    }

    ifos = ["H1", "L1"]
    tensors, vertices = gw.get_ifo_geometry(*ifos)

    responses = gw.compute_observed_strain(
        dec,
        psi,
        phi,
        detector_tensors=tensors,
        detector_vertices=vertices,
        sample_rate=SAMPLE_RATE,
        **polarizations,
    )

    background_samples = generate_background(
        background_file,
        num_samples=len(signals),
        save_file=False,
    )

    # Place coalescence point of the signal at 10 seconds into the kernel
    signal_time = 10
    signal_idx = int(signal_time * SAMPLE_RATE)

    # Crop and/or pad responses to match the length of the background samples
    if coalescence_idx >= signal_idx:
        responses = responses[:, :, coalescence_idx - signal_idx :]
    else:
        responses = np.pad(
            responses,
            ((0, 0), (0, 0), (signal_idx - coalescence_idx, 0)),
            mode="constant",
            constant_values=0,
        )

    if responses.shape[-1] > background_samples.shape[-1]:
        responses = responses[:, :, : background_samples.shape[-1]]
    else:
        responses = np.pad(
            responses,
            ((0, 0), (0, 0), (0, background_samples.shape[-1] - responses.shape[-1])),
            mode="constant",
            constant_values=0,
        )

    injection_dataset = background_samples + responses
    
    output_file = waveform_file.replace(".hdf5", "_injections.hdf5")
    with h5py.File(output_file, "w") as f:
        f.create_dataset("data", data=injection_dataset)


def main(
    background_file: str,
    waveform_files: List[str],
    kernel_length: float,
    num_samples: int,
) -> None:
    print("Generating background samples")
    _ = generate_background(background_file, kernel_length, num_samples)
    for waveform_file in waveform_files:
        print(f"Generating injections for {waveform_file}")
        generate_injections(waveform_file, background_file)

if __name__ == "__main__":
    main(
        background_file="",
        waveform_files=[""],
        kernel_length=10.5,
        num_samples=100000,
    )