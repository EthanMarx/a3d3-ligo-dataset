
import astropy.cosmology as cosmo
import numpy as np
from bilby.core.prior import (
    Cosine,
    PriorDict,
    Uniform,
)
from bilby.gw.prior import UniformSourceFrame

from priors.utils import mass_constraints


# default cosmology
COSMOLOGY = cosmo.Planck15

def prior(
    mass_1: float,
    mass_2: float,
):

    prior = PriorDict(conversion_function=mass_constraints)
    prior["mass_1"] = mass_1
    prior["mass_2"] = mass_2
    prior["redshift"] = UniformSourceFrame(
        name="redshift", minimum=0, maximum=2
    )
    prior["dec"] = Cosine(name="dec")
    prior["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    detector_frame_prior = False
    return prior, detector_frame_prior