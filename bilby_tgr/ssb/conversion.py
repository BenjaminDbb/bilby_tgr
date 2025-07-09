import astropy.units as u
from astropy.cosmology import z_at_value
from bilby.gw.cosmology import get_cosmology
import bilby
from scipy import integrate
import numpy as np


def isiterable(obj):
    """
    Returns True if the given object is iterable.
    Copied from astropy, as location varies between versions
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def vectorize_if_needed(func, *x):
    """
    Helper function to vectorize functions on array inputs
    Copied here from astropy, because vectorization differs between astropy versions
    """
    if any(map(isiterable, x)):
        return np.vectorize(func)(*x)
    else:
        return func(*x)


def luminosity_distance_to_redshift_scalar(distance, cosmology=None):
    """
    copied from bilby, but non-vectorized
    """
    cosmology = get_cosmology(cosmology)

    return z_at_value(cosmology.luminosity_distance, distance * u.Mpc).value


def kv5_from_ssb_coeff_scalar(ssb_coeff, z, cosmology=None):
    """
    converts modified dispersion coefficient to the SME parameter kv5_00 as
    in Eq. 13 of https://arxiv.org/pdf/2210.04481.pdf

    Parameters:

    ssb_coeff: float
        modified dispersion coefficient [unit: m*s]
    z: float
        redshift
    cosmology: astropy.cosmology.FLRW, str, dict
        Description of cosmology, one of:
        None - Use DEFAULT_COSMOLOGY from bilby
        Instance of astropy.cosmology.FLRW subclass
        String with name of known Astropy cosmology, e.g., Planck13
        Dictionary with arguments required to instantiate the cosmology
        class.

    Returns:

    kv5:  float
        SME parameter [unit: m]
    """
    cosmo = get_cosmology(cosmology)

    # compute tau
    def tau_fct(z):
        return (1 + z) / (cosmo.H(z) / u.Mpc.to(u.km) * u.megaparsec).value
    tau = integrate.quad(tau_fct, 0, z)

    # convert effective parameter
    kv5 = ssb_coeff / tau[0]

    return kv5


def ssb_coeff_from_kv5_scalar(kv5, z, cosmology=None):
    """
    The inverse of ssb_coeff_from_kv5_scalar

    Parameters:

    kv5:  float
        SME parameter [unit: m]
    z: float
        redshift
    cosmology: astropy.cosmology.FLRW, str, dict
        Description of cosmology, one of:
        None - Use DEFAULT_COSMOLOGY from bilby
        Instance of astropy.cosmology.FLRW subclass
        String with name of known Astropy cosmology, e.g., Planck13
        Dictionary with arguments required to instantiate the cosmology
        class.

    Returns:

    ssb_coeff: float
        modified dispersion coefficient [unit: m*s]
    """
    cosmo = get_cosmology(cosmology)

    # compute tau
    def tau_fct(z):
        return (1 + z) / (cosmo.H(z) / u.Mpc.to(u.km) * u.megaparsec).value
    tau = integrate.quad(tau_fct, 0, z)

    # convert effective parameter
    ssb_coeff = kv5 * tau[0]

    return ssb_coeff


def luminosity_distance_to_redshift(distance, cosmology=None):
    """
    copied from bilby, but non-vectorized
    """
    if isiterable(distance):
        distance = np.asarray(distance)
    return vectorize_if_needed(luminosity_distance_to_redshift_scalar, distance, cosmology)


def kv5_from_ssb_coeff(ssb_coeff, z, cosmology=None):
    """
    vectorized version of kv5_from_ssb_coeff_scalar
    """
    if isiterable(z):
        z = np.asarray(z)
    if isiterable(ssb_coeff):
        ssb_coeff = np.asarray(ssb_coeff)
    return vectorize_if_needed(kv5_from_ssb_coeff_scalar, ssb_coeff, z, cosmology)


def ssb_coeff_from_kv5(kv5, z, cosmology=None):
    """
    vectorized version of ssb_coeff_from_kv5_scalar
    """
    if isiterable(z):
        z = np.asarray(z)
    if isiterable(kv5):
        kv5 = np.asarray(kv5)
    return vectorize_if_needed(ssb_coeff_from_kv5_scalar, kv5, z, cosmology)


def generate_ssb_parameters(sample, cosmology=None):

    output_sample = sample.copy()
    redshift = None
    if 'redshift' in output_sample:
        redshift = output_sample['redshift']
    elif redshift is None:
        redshift = bilby.gw.conversion.luminosity_distance_to_redshift(output_sample['luminosity_distance'], cosmology)
    else:
        print('Check the parameters. There is neither luminosity distance nor redshift.')

    if 'ssb_coeff' in output_sample:
        output_sample['kv5'] = kv5_from_ssb_coeff(output_sample['ssb_coeff'], redshift, cosmology=cosmology)
    elif 'kv5' in output_sample:
        output_sample['ssb_coeff'] = ssb_coeff_from_kv5(output_sample['kv5'], redshift, cosmology=cosmology)
    else :
        print('could not generate samples of ssb_coeff or kv5')

    return output_sample


def generate_all_bbh_parameters(sample, likelihood=None, priors=None, npool=1):
    """
    adapted from bilby
    From either a single sample or a set of samples fill in all missing
    BBH parameters, in place.

    Parameters
    ==========
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an
        injection or posterior samples.
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient, optional
        GravitationalWaveTransient used for sampling, used for waveform and
        likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.
    """
    output_sample = bilby.gw.conversion.generate_all_bbh_parameters(sample, likelihood, priors, npool)
    output_sample = generate_ssb_parameters(output_sample)
    return output_sample


def convert_to_lal_binary_black_hole_parameters(parameters):
    """
    Convert parameters we have into parameters we need.

    This is defined by the parameters of bilby.source.lal_binary_black_hole()


    Mass: mass_1, mass_2
    Spin: a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl
    Extrinsic: luminosity_distance, theta_jn, phase, ra, dec, geocent_time, psi
    SSB: ssb_coeff

    This involves popping a lot of things from parameters.
    The keys in added_keys should be popped after evaluating the waveform.

    Parameters
    ==========
    parameters: dict
        dictionary of parameter values to convert into the required parameters

    Returns
    =======
    converted_parameters: dict
        dict of the required parameters
    added_keys: list
        keys which are added to parameters during function call
    """

    converted_parameters, added_keys = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(parameters)
    if 'ssb_coeff' not in converted_parameters:
        if 'kv5' in converted_parameters:       # for the isotropic limit
            redshift = bilby.gw.conversion.luminosity_distance_to_redshift(converted_parameters['luminosity_distance'])
            converted_parameters['ssb_coeff'] = ssb_coeff_from_kv5(converted_parameters['kv5'], redshift)

        added_keys.append('ssb_coeff')

    return converted_parameters, added_keys
