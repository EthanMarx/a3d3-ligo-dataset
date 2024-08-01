import os

import law
import luigi
from luigi.util import inherits

from aframe.base import AframeWrapperTask
from aframe.tasks import Fetch, TrainingWaveforms
from aframe.tasks.data.base import AframeDataTask
from aframe.parameters import PathParameter

class config(luigi.Config):
    base_dir = PathParameter(default=os.getenv("AFRAME_BASE_DIR"))
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condor_dir = self.base_dir / "condor"
        self.data_dir = self.base_dir / "data"



class Background(AframeDataTask):
    """
    Fetch background strain and generate kernels of background
    samples from timesliding strain
    """
    sample_rate = luigi.FloatParameter()
    num_background_samples = luigi.IntParameter()
    kernel_length = luigi.FloatParameter()

    def requires(self):
        return Fetch.req(
            self,
            data_dir = config().data_dir / "background",
            segments_file = config().data_dir / "background" / "segments.txt",
            condor_directory = config().condor_dir / "fetch",
        )

    def output(self):
        return law.LocalFileTarget(config().data_dir / "background.hdf5")

    def run(self):
        from a3d3_ligo_dataset.generate_datasets import generate_background
        import h5py
        background_file = list(self.input().collection.targets.values())[0].path
        background = generate_background(
            background_file,
            self.kernel_length,
            self.num_background_samples,
            self.sample_rate,
        )

        with h5py.File(self.output().path, "w") as f:
            f.create_dataset("data", data=background)

class Injections(AframeDataTask):
    mass_pairs = luigi.ListParameter()
    zmax = luigi.FloatParameter()
    prior = luigi.Parameter()
    sample_rate = luigi.FloatParameter()
    kernel_length = luigi.FloatParameter()

    @property
    def default_image(self):
        return "data.sif"

    def output(self):
        outputs = {}
        for (m1, m2) in self.mass_pairs:
            outputs[(m1, m2)] = law.LocalFileTarget(config().data_dir / f"./waveforms_{m1}_{m2}.hdf5")
        return outputs

    def requires(self):
        reqs = {}
        reqs["background"] = Fetch.req(
            self,
            data_dir = config().data_dir / "background",
            segments_file = config().data_dir / "background" / "segments.txt",
            condor_directory = config().condor_dir / "fetch",
        )
        for (m1, m2) in self.mass_pairs:
            prior_args = {"mass_1": m1, "mass_2": m2, "zmax": self.zmax}
            reqs[(m1, m2)] = TrainingWaveforms.req(
                self,
                condor_directory = config().condor_dir / f"waveforms_{m1}_{m2}",
                output_dir = config().data_dir / f"waveforms_{m1}_{m2}",
                prior=self.prior,
                prior_args=prior_args
            )
        return reqs


    def run(self):
        from a3d3_ligo_dataset.generate_datasets import generate_injections
        import h5py

        background_file = list(self.input()["background"].collection.targets.values())[0].path
        for (m1, m2) in self.mass_pairs:
            waveform_file = list(self.input()[(m1, m2)].collection.targets.values())[0].path
            output_file = self.output()[(m1, m2)].path
            injections, snrs = generate_injections(
                waveform_file,
                background_file,
                self.sample_rate,
                self.kernel_length,
            )

            with h5py.File(output_file, "w") as f:
                f.create_dataset("data", data=injections)
                f.create_dataset("snrs", data=snrs)


@inherits(Background, Injections)
class A3D3Dataset(AframeWrapperTask):
    def requires(self):
        yield Background.req(self)
        yield Injections.req(self)




