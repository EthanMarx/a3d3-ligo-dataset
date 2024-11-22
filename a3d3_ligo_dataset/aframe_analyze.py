from pathlib import Path
from typing import List

import h5py
import torch
from utils.preprocessing import BatchWhitener
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchaudio.transforms import Resample

class MultiRateResample(torch.nn.Module):
    """
    Resample a time series to multiple different sample rates

    Args:
        original_sample_rate:
            The sample rate of the original time series in Hz
        duration:
            The duration of the original time series in seconds
        new_sample_rates:
            A list of new sample rates that different portions
            of the time series will be resampled to
        breakpoints:
            The time at which there is a transition from one
            sample rate to another

    Returns:
        A time series Tensor with each of the resampled segments
        concatenated together
    """

    def __init__(
        self,
        original_sample_rate: int,
        duration: float,
        new_sample_rates: List[int],
        breakpoints: List[float],
    ):
        super().__init__()
        self.original_sample_rate = original_sample_rate
        self.duration = duration
        self.new_sample_rates = new_sample_rates
        self.breakpoints = breakpoints
        self._validate_inputs()

        # Add endpoints to breakpoint list
        self.breakpoints.append(duration)
        self.breakpoints.insert(0, 0)

        self.resamplers = torch.nn.ModuleList(
            [Resample(original_sample_rate, new) for new in new_sample_rates]
        )
        idxs = [
            [int(breakpoints[i] * new), int(breakpoints[i + 1] * new)]
            for i, new in enumerate(self.new_sample_rates)
        ]
        self.register_buffer("idxs", torch.Tensor(idxs).int())

    def _validate_inputs(self):
        if len(self.new_sample_rates) != len(self.breakpoints) + 1:
            raise ValueError(
                "There are too many/few breakpoints given "
                "for the number of frequencies"
            )
        if max(self.breakpoints) >= self.duration:
            raise ValueError(
                "At least one breakpoint was greater than the given duration"
            )
        if not self.breakpoints[1:] > self.breakpoints[:-1]:
            raise ValueError("Breakpoints must be sorted in ascending order")

    def forward(self, X: torch.Tensor):
        X = X.contiguous()
        segments = []
        for i in range(len(self.resamplers)):
            resampler = self.resamplers[i]
            idx = self.idxs[i]
            segment = resampler(X)[..., idx[0] : idx[1]]
            segments.append(segment)
        return torch.cat(segments, dim=-1)
            

def load_data(data_file: str):
    with h5py.File(data_file, "r") as f:
        data = f["data"][:]
        try:
            snrs = f["snrs"][:]
        except KeyError:
            snrs = None
    return data, snrs

def load_model(weights_file: str):
    return torch.jit.load(weights_file)

def infer(
    data_file: Path,
    model,
    whitener: torch.nn.Module,
):
    print(f"Loading data from {data_file.name}")
    data, snrs = load_data(data_file)
    data = torch.Tensor(data)

    outputs = []
    inf_batch_size = 4
    with torch.no_grad():
        for i in tqdm(range(len(data) // inf_batch_size)):
            sample = data[inf_batch_size * i : inf_batch_size * (i + 1)]
            whitened = whitener(sample.to("cuda"))
            output = model(whitened)
            outputs.append(output.to("cpu"))
            

    outputs = torch.concatenate(outputs)
    outputs = outputs.squeeze()

    return outputs, snrs

def main(
    data_files: List[Path],
    weights_file: str,
    output_file: str,
    device="cuda",
):

    resampler = MultiRateResample(
        original_sample_rate = 2048,
        duration = 6,
        new_sample_rates = [128, 256, 512, 1024, 2048],
        breakpoints = [4., 5., 5.5, 5.75]
    ).to(device)

    whitener = BatchWhitener(
        kernel_length=6,
        sample_rate=2048,
        inference_sampling_rate=2048,
        batch_size=1,
        fduration=2,
        fftlength=8,
        highpass=32,
        augmentor = resampler
    ).to(device)
    model = load_model(weights_file=weights_file).to(device)

    with h5py.File(output_file, "w") as f:
        for file in data_files:
            outputs, snrs = infer(file, model, whitener)
            g = f.create_group(file.stem)
            g.create_dataset(file.stem, data=outputs)
            if snrs is not None:
                g.create_dataset("snrs", data=snrs)

if __name__ == "__main__":
    data_path = Path("/home/ethan.marx/projects/a3d3-ligo-dataset/data/multi-rate-data/data")
    data_files = [file for file in data_path.iterdir() if not file.is_dir()]
    main(
        data_files=data_files,
        weights_file="/home/william.benoit/aframe/runs/mult-rate-resampling/training/model.pt",
        output_file="/home/ethan.marx/projects/a3d3-ligo-dataset/multi-rate-debug.hdf5",
    )


