import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
from OOPAO.calibration.CalibrationVault import CalibrationVault

def InteractionMatrix_test(ngs,tel,dm,wfs,M2C,stroke,phaseOffset=0,nMeasurements=50,
                           noise='off',invert=True,print_time=False,display=False,display_slopes = False,single_pass=True):
    """

    This function allows to build the Interaction Matrix (or its inverse if invert == True) of an AO system.
    IntMat is a 2D matrix with DM modes columns and wfs.nSignal lines. Each column of the IntMat is the measured signal
    by the wfs when applying a certain mode in the DM.

    :param ngs: Source object
    :param atm: Atmosphere object
    :param tel: Telescope object
    :param dm: Deformable mirror object
    :param wfs: Wavefront sensor object
    :param M2C: Mode to command matrix: can be either zonal or modal
    :param stroke: stroke for the push/pull in M2C units
    :param phaseOffset: phase offset considered when computing the Interaction matrix. Can be useful to take into account possible
                         non-common path aberrations (NCPAs) for instance.
    :param nMeasurements: number of simultaneous wfs measurements. Instead of building the interaction matrix column by column, we do it in chunks
                          of nMeasurements columns.
    :param noise: if noise = "off" (default), we do not keep the noise when doing the wfs measurements
    :param invert: boolean variable. If True, we invert the Interaction matrix. Otherwise, the output is the Interaction matrix itself
    :param print_time: boolean variable: if True, we print the elapsed time in seconds per iteration when building the IntMat. Default is False
    :param display: boolean variable: if True (default), we display time using tqdm module
    :param single_pass: boolean variable: if True (default), we only perform push. If False, we perform push and pull

    :return: InteractionMatrix (intMat) if invert == False or the inverse of the InteractionMatrix if invert == True

    """

    if display is False:
        def iterate(x):
            return x
    else:
        def iterate(x):
            return tqdm.tqdm(x)

    if wfs.tag == 'pyramid' and wfs.gpu_available:
        nMeasurements = 1
        print('Pyramid with GPU detected => using single mode measurement to increase speed.')

    # disabled noise functionality from WFS
    if noise == 'off':
        wfs.cam.photonNoise = 0
        wfs.cam.readoutNoise = 0
        wfs.cam.backgroundNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')

        # separate tel from ATM
    tel.isPaired = False
    ngs * tel

    try:
        nModes = M2C.shape[1]
    except:
        nModes = 1

    intMat2D = np.zeros([wfs.signal_2D.shape[0]*wfs.signal_2D.shape[1], nModes])
    intMat = np.zeros([wfs.nSignal, nModes])
    nCycle = int(np.ceil(nModes / nMeasurements))
    nExtra = int(nModes % nMeasurements)

    if nMeasurements > nModes:
        nMeasurements = nModes

    if np.ndim(phaseOffset) == 2:
        if nMeasurements != 1:
            phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

    for i in iterate(range(nCycle)):

        if nModes > 1:
            if i == nCycle - 1:
                if nExtra != 0:
                    intMatCommands = np.squeeze(M2C[:, -nExtra:])
                    try:
                        phaseBuffer = np.tile(phaseOffset[..., None], (1, 1, intMatCommands.shape[-1]))
                    except:
                        phaseBuffer = phaseOffset
                else:
                    intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
            else:
                intMatCommands = np.squeeze(M2C[:, i * nMeasurements:((i + 1) * nMeasurements)])
        else:
            intMatCommands = np.squeeze(M2C)

        a = time.time()
        # push
        dm.coefs = intMatCommands * stroke
        tel * dm
        tel.src.phase += phaseBuffer
        tel.src.phase_no_pupil += phaseBuffer  # this needed when using the geometric SH
        tel * wfs
        sp = wfs.signal
        if display_slopes and i==1:
            sig2D = wfs.signal_2D
            plt.figure(), plt.imshow(sig2D[0,:,:]),plt.title('push'), plt.colorbar(),plt.show(block=False)

        # pull
        if single_pass:
            sm = 0 * wfs.signal
            factor = 2
        else:
            dm.coefs = -intMatCommands * stroke
            tel * dm
            tel.src.phase += phaseBuffer
            tel.src.phase_no_pupil += phaseBuffer  # this needed when using the geometric SH
            tel * wfs
            sm = wfs.signal
            factor = 1
            if display_slopes and i==1:
                sig2D = wfs.signal_2D
                plt.figure(), plt.imshow(sig2D[0,:,:]),plt.title('pull'), plt.colorbar(),plt.show(block=False)


        if sp.shape[0] != wfs.nSignal:
            sp = sp.transpose()
            sm = sm.transpose()

        if i == nCycle - 1:
            if nExtra != 0:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    if nExtra == 1:
                        intMat[:, -nExtra] = np.squeeze(0.5 * (sp - sm) / stroke)
                    else:
                        intMat[:, -nExtra:] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                if nMeasurements == 1:
                    intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
                else:
                    intMat[:, -nMeasurements:] = np.squeeze(0.5 * (sp - sm) / stroke)
        else:
            if nMeasurements == 1:
                intMat[:, i] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                intMat[:, i * nMeasurements:((i + 1) * nMeasurements)] = np.squeeze(0.5 * (sp - sm) / stroke)

        intMat = np.squeeze(intMat)

        if print_time:
            print(str((i + 1) * nMeasurements) + '/' + str(nModes))
            b = time.time()
            print('Time elapsed: ' + str(b - a) + ' s')

    out = CalibrationVault(factor * intMat, invert=invert)

    return out

