import law
import luigi


from aframe.tasks import Fetch, TrainingWaveforms
from aframe.base import AframeParameters


class A3D3Dataset(AframeParameters):
    mass_pairs = luigi.ListParameter()
    prior = luigi.Parameter()
    sample_rate = luigi.FloatParameter()
    kernel_length = luigi.FloatParameter()
    num_waveform_samples = luigi.IntParameter()
    num_background_samples = luigi.IntParameter()

    def output(self):
        outputs = {}
        for (m1, m2) in self.mass_pairs:
            outputs[(m1, m2)] = law.LocalFileTarget(f"waveforms_{m1}_{m2}.hdf5")
        outputs["background"] = law.LocalFileTarget("background.hdf5")
        return outputs
        
    def requires(self):
        reqs = {}
        for (m1, m2) in self.mass_pairs:
            prior_args = {"mass_1": m1, "mass_2": m2}
            reqs[(m1, m2)] = TrainingWaveforms.req(
                self,
                prior=self.prior, 
                prior_args=prior_args
            )
        reqs["background"] = Fetch.req(self)
        return reqs

    def run(self):
        from a3d3_ligo_dataset.generate_datasets import generate_background, generate_injections
        background_file = self.input()["background"][0].path
        generate_background(
            background_file, 
            self.kernel_length, 
            self.num_background_samples,
            self.sample_rate,
            self.output()["background"].path
        )

        for (m1, m2), waveform_file in self.input().items():
            output_file = self.output()[(m1, m2)].path
            generate_injections(
                waveform_file.path, 
                background_file,
                self.sample_rate,
                output_file
            )


