from pathlib import Path
from typing import List

import h5py
import torch
from utils.preprocessing import BatchWhitener
from tqdm import tqdm


def load_data(data_file: str):
    with h5py.File(data_file, "r") as f:
        data = f["data"][:]
    return data

def load_model(weights_file: str):
    return torch.jit.load(weights_file)

def infer(
    data_file: Path,
    model,
    whitener: torch.nn.Module,
):
    print(f"Loading data from {data_file.name}")
    data = torch.Tensor(load_data(data_file))

    outputs = []
    inf_batch_size = 1000
    with torch.no_grad():
        for i in tqdm(range(len(data) // inf_batch_size)):
            sample = data[inf_batch_size * i : inf_batch_size * (i + 1)]
            outputs.append(model(whitener(sample.to("cuda"))).to("cpu"))

    outputs = torch.concatenate(outputs)
    outputs = outputs.squeeze()

    return outputs

def main(
    data_files: List[Path],
    weights_file: str,
    output_file: str,
    device="cuda"
):
    whitener = BatchWhitener(
        kernel_length=1.5,
        sample_rate=2048,
        inference_sampling_rate=2048,
        batch_size=1,
        fduration=1,
        fftlength=2.5,
        highpass=32,
    ).to(device)
    model = load_model(weights_file=weights_file).to(device)

    with h5py.File(output_file, "w") as f:
        for file in data_files:
            outputs = infer(file, model, whitener)
            f.create_dataset(file.stem, data=outputs)

if __name__ == "__main__":
    data_path = Path("../data/data")
    data_files = [file for file in data_path.iterdir() if not file.is_dir()]
    main(
        data_files=data_files,
        weights_file="../aframe_weights.pt",
        output_file="../submission.hdf5",
    )


