
import argparse
import sys

import numpy as np
import scipy.fftpack as fft
# import scipack
from PIL import Image
from skimage.transform import rotate

verbose = False

def show_progress(progress):
    """
    Shows the progress as a bar.
    Adapted from:
    https://stackoverflow.com/questions/3160699/python-progress-bar#3160819
    """
    barLength = 50
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1}%".format("#"*block + "-"*(barLength-block), np.floor(progress*100))
    sys.stdout.write(text)
    sys.stdout.flush()


def no_filter(ffts):
    f = np.array([0.0] + [1/360.]*(ffts.shape[1] - 1))
    return ffts * f


def ramp_filter(ffts):
    """
    Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ffts: result of the fourier transform
    """
    # ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    ramp = np.abs(np.fft.fftfreq(int(ffts.shape[1])))
    return ffts * ramp


def hamming_windowed_ramp_filter(ffts):
    """
    Filter a 2-d array of 1-d FFTs using a hamming windowed ramp filter.
    ffts: result of the fourier transform
    """
    ramp = np.abs(np.fft.fftfreq(int(ffts.shape[1])))
    # m = len(ramp) // 2
    # for i in range(1, len(ramp)):
        # ramp[i] *= (0.54 + 0.46 * np.cos(np.pi * (i - m) / m ))
    ramp[1:] = ramp[1:] * (0.54 + 0.46 * np.cos(ramp[1:]))
    # print(ramp)
    return ffts * ramp


def backproject(iffts):
    """
    Perform the backprojection i.e. sum up the rotated iffts.
    iffts: real parts of the inverse fourier transform
    """
    l = iffts.shape[1]
    result = np.zeros((l, l))
    steps = iffts.shape[0]
    angles = np.linspace(0, 180, steps, endpoint=False)
    for i, theta in enumerate(angles):
        if verbose:
            show_progress(i / (steps - 1))
        tmp = np.array([iffts[i], ] * l)
        rotated = rotate(tmp, theta)
        result += rotated
    return result


def reconstruct(image, filter_func):
    """
    Reconstructs the CT scan image from the input image.
    image: 2d array containing the images grey scale values
    filter_func: function used for filtering
    """
    # 0. Calculate ffts
    if verbose:
        print("Step 0: Calculating ffts...")
    ffts = fft.fft(image)
    # 1. Filter
    if verbose:
        print("Step 1: Filtering image...")
    filtered = filter_func(ffts)

    # 2. calculate ifft
    if verbose:
        print("Step 2: Calculating iffts...")
    ifft = np.real(fft.ifft(filtered))

    # 3. backprojecting
    if verbose:
        print("Step 3: Projecting / summing up...")
    recon = backproject(ifft)
    if verbose:
        print()
    return recon


def main():
    """
    Main function.
    """
    global verbose
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", type=str, default="sinogram.png",
                        help="File to use for the reconsonstruction.")
    parser.add_argument("--filter", type=str, default="none",
                        help="Choose filter type: Possible values:\n"
                        "none: no filter \n"
                        "ramp: ramp filtering \n"
                        "ham: hamming windowed ramp filtering")
    parser.add_argument("-v", action="store_true",
                        help="Verbose output.")
    args = parser.parse_args()
    verbose = args.v
    # load image into 2d array
    image = np.asarray(Image.open(args.file).convert("L"))
    if verbose:
        print("Read {0}x{1} image.".format(*image.shape))
    # determine filter
    if args.filter == "none":
        fil = no_filter
    elif args.filter == "ramp":
        fil = ramp_filter
    elif args.filter == "ham":
        fil = hamming_windowed_ramp_filter
    else:
        print("Invalid filter type.")
        return
    if verbose:
        print("Reconstructing using {}.".format(fil.__name__.replace("_", " ")))
    # reconstruct the original image using the filter
    recon = reconstruct(image, fil)
    # display the image
    if verbose:
        print("Displaying image.")
    im = Image.fromarray(recon)
    im.show()

if __name__ == "__main__":
    main()
