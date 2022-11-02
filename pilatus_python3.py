# Calibration and integration routines for SSRL 2-1
# Beam center calculation and Pilatus 100K integration
# Written by Kevin Stone
# Refactored for python3 by Yue Wu

import os
import math
import gc

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate
import lmfit

from tqdm import tqdm


def find_directbeam(data_path, calib_name, plot=True):
    '''
    Find the direct beam position in the image
    Return the direct beam pixel as a list
    '''
    i = 50
    filename = data_path + calib_name + str(i).zfill(4) + ".raw"
    Z = read_RAW(filename)
    # plt.imshow(data, aspect='auto')

    x = np.linspace(1, np.shape(Z)[1], np.shape(Z)[1])
    y = np.linspace(1, np.shape(Z)[0], np.shape(Z)[0])
    Xg, Yg = np.meshgrid(x, y)

    model = lmfit.models.Gaussian2dModel()
    params = model.guess(Z.flatten(), Xg.flatten(), Yg.flatten())  # method cannot deal with 2D data
    result = model.fit(Z, x=Xg, y=Yg, params=params)
    # lmfit.report_fit(result) # print the fit result

    centerx = int(result.best_values['centerx'])
    centery = int(result.best_values['centery'])
    print(f"Found beam center at ({centerx}, {centery})")
    
    if plot:
        # Plot the beam center
        plt.imshow(Z, aspect='auto')
        plt.plot(centerx, centery, 'w+', markersize=30, alpha=0.5)
        annotation = f'Beam Center \n ({centerx}, {centery})' 
        plt.annotate(annotation, color='w', xy=(centerx, centery), xytext=(centerx+15, centery+15))
        plt.show()
    
    return [centerx, centery]

def read_TIFF(file):
    '''
    Read in a TIFF file and return the image data as a numpy array
    '''
    # print("Reading TIFF file here...")
    try:
        im = open(file, 'rb')
        im.seek(4096)	# skip the first 4096 bytes of header info for TIFF images
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        #arr = np.fliplr(arr)  #for the way mounted at BL2-1
        print(np.shape(arr))
        print(len(arr))
        return arr
    except:
        print(f"Error reading file: {file}")
        return None

def read_RAW(file):
    '''
    Read in a RAW file and return the image data as a numpy array
    '''
    # print("Reading RAW file here...")
    try:
        im = open(file, 'rb')
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        arr.shape = (195, 487)
        #arr = np.fliplr(arr)  #for the way mounted at BL2-1
        return arr
    except:
        print(f"Error reading file: {file}")
        return None

def csvread(filename):  
    # print( "Reading CSV file here...")
    csv = open(filename)
    line = csv.readline()
    temp = line.split(',')
    xi = 1
    yi = 4
    i0i = 3
    x = []
    y = []
    i0 = []
    line = csv.readline()
    while line:
        temp = line.split(",")
        x = np.append(x, float(temp[xi]))
        y = np.append(y, float(temp[yi]))       
        i0 = np.append(i0, float(temp[i0i]))
        line = csv.readline()
    csv.close()
    return x, y, i0

def gauss_linbkg(x, m, b, x0, intint, fwhm):
    return m*x + b + intint*(2./fwhm)*np.sqrt(np.log(2.)/np.pi)*np.exp(-4.*np.log(2.)*((x-x0)/fwhm)**2)
    
def Gauss_fit(x, y):
    pguess = [0, 0, np.argmax(y), np.max(y), 5.0]  # linear background (2), pos, intensity, fwhm
    popt, pcov = curve_fit(gauss_linbkg, x, y, p0=pguess)
    return popt
    
def simple_line(x, m, b):
    return m*x + b

def directbeam_calibration(csv_path, csv_name, data_path, calib_name, db_pixel, pix_size=172.0, plot=True):
    '''
    This function determines the sample to detector distance.
    '''
    # Read CSV file, get step size and number of points in scan
    x, y, i0 = csvread(os.path.join(csv_path, csv_name))

    num_points = len(x)
    #calib_tth_steps = (x[-1] - x[0])/num_points
    calib_tth_steps = abs(x[1] - x[0])
    x = []
    y = []
    i0 = []

    # Read images, take line cut, fit peak for Al2O3 calibration scan
    pks = []
    
    for i in tqdm(range(1, num_points)):
        # print(i)
        filename = data_path + calib_name + str(i).zfill(4) + ".raw"
        data = read_RAW(filename)
        x = np.arange(0, np.shape(data)[1])
        y = data[db_pixel[1], :]
        y += data[db_pixel[1] + 1, :]
        y += data[db_pixel[1] - 1, :]
        y += data[db_pixel[1] + 2, :]
        y += data[db_pixel[1] - 2, :]
        y += data[db_pixel[1] + 3, :]
        y += data[db_pixel[1] - 3, :]
        y += data[db_pixel[1] + 4, :]
        y += data[db_pixel[1] - 4, :]
        popt = Gauss_fit(x, y)
        # # This plots each fit (lots of plots)
        # if plot:
            # plt.figure()
            # plt.cla()
            # plt.plot(x,y, 'b.')
            # plt.plot(x, gauss_linbkg(x, *popt), 'r-')
        pks = np.append(pks, popt[2])

    # Fit line to the extracted peak positions and determine the sample to detector distance
    x = np.arange(1,num_points)
    lin_fit, pcov = curve_fit(simple_line, pks, x*calib_tth_steps + 0.00)
    det_R = 1.0/np.tan(abs(lin_fit[0])*np.pi/180.0)     # sample to detector distance in pixels
    
    # print("Sample to detector distance in pixels = " + str(det_R))
    print(f"Sample to detector distance {det_R:.2f} pixels or {((det_R * pix_size / 1000.0)):.2f} mm")
    
    if plot:
        plt.figure()
        plt.plot(pks, x*calib_tth_steps, 'b.')
        plt.plot(pks, lin_fit[0]*pks + lin_fit[1], 'r-')
        plt.show()

    outname = csv_path + csv_name[:-4] + "_calib.cal"
    outfile = open(outname, "w")
    outfile.write("direct_beam_x \t %i\n"  % db_pixel[0])
    outfile.write("direct_beam_y \t %i\n" % db_pixel[1])
    outfile.write("Sample_Detector_distance_pixels \t %15.6G\n" % det_R)
    outfile.write("Sample_Detector_distance_mm \t %15.6G" % (det_R * pix_size / 1000.0))
    outfile.close()

    return (det_R, det_R * pix_size / 1000.0)

def SPECread(filename,scan_number):
    '''
    Read in a SPEC file and return tth and i0 data as numpy arrays
    '''
    print(f"Reading SPEC file {filename}")
    tth = []
    i0 = []
    spec = open(filename)
    for line in spec:
        if "#O" in line and "tth" in line:  #find which line has the 2theta position
            temp = line.split()
            tth_line = temp[0][2]
            for i in range(0, len(temp)):
                if temp[i] == "tth":	#find where in that line the 2theta position is listed
                    tth_pos = i
                    break
            break
    for line in spec:
        if "#S" in line:
            temp = line.split()
            if int(temp[1]) == scan_number:
                break
    for line in spec:
        if "#P" + str(tth_line) in line:
            temp = line.split()
            tth_start = float(temp[tth_pos])
            break
    for line in spec:
        if "#L" in line:
            motors = line.split()[1:]
            if "tth" not in line:
                tth_motor_bool = False
                print("2theta is not scanned...")
            else:
                tth_motor_bool = True
                tth_motor = motors.index("tth")
            i0_motor = motors.index("Monitor")
            break
    for line in spec:
        try:
            temp = line.split()
            if tth_motor_bool:
                tth = np.append(tth, float(temp[tth_motor]))
            else:
                tth = np.append(tth, tth_start)
            i0 = np.append(i0, float(temp[i0_motor]))
        except:
            break
    spec.close()
    return tth, i0

def rotate_operation(map, tth):
    # Apply a rotation operator to map this into cartesian coordinates (x',y',z') when 2-theta != 0
    # We should be efficient about how this is implemented
    xyz_map_prime = np.empty_like(map)
    tth *= np.pi/180.0
    rot_op = np.array([[1.0, 0.0, 0.0], 
                       [0.0, np.cos(tth), np.sin(tth)], 
                       [0.0, -1.0*np.sin(tth), np.cos(tth)]])
    #print(np.shape(data))
    xyz_map_prime = np.matmul(rot_op, map)
    return xyz_map_prime
        
def cart2sphere(map, data):
    # Convert the rotated cartesian coordinate map to spherical coordinates
    # This should also be efficiently implemented
    tth_map = np.empty_like(data, dtype=float).flatten()
    _r = np.sqrt((map[:2,:]**2).sum(axis=0))
    tth_map = np.arctan(_r/map[2, :])*180.0/np.pi
    tth_map = tth_map.reshape(data.shape)
    return tth_map%180.0

def integrate(scan_number, image_path, spec_path, user, spec_name, stepsize, xyz_map, mult=1000000.0, plot_integration = True, plot_binning = False):
    '''
    Generate 1D data from 2D frames
    '''
    #image_path = folder + "images/"
    #spec_path = folder + "scans/"

    if np.isscalar(scan_number):
        scan_number = [scan_number]
        # print("Converting scan number to array...")
    for scan_num in scan_number:
        # print(scan_num)
        tth, i0 = SPECread(spec_path + spec_name, scan_num)
        x = []
        y = []
        xmax_global = 0.0  #set this to some small number so that it gets reset with the first image
        xmin_global = 180.0
        bins = np.arange(0.0, 180.0, stepsize)
        digit_y = np.zeros_like(bins)
        digit_norm = np.zeros_like(bins)
        for k in tqdm(range(0, len(tth))):
            x = []
            y = []
            #print k
            # print(tth[k])
            filename = image_path + user + "_" + spec_name + "_scan" + str(scan_num) + "_" + str(k).zfill(4) + ".raw"
            # print(filename)
            data = read_RAW(filename)
            xyz_map_prime = rotate_operation(xyz_map, tth[k])
            tth_map = cart2sphere(xyz_map_prime, data)
            x = tth_map.flatten()
            y = data.flatten()/i0[k]
            xmax = np.max(x)
            xmin = np.min(x)
            xmax_global = np.max([xmax, xmax_global])
            xmin_global = np.min([xmin, xmin_global])
            sort_index = np.argsort(x)
            y_0 = np.where(y < 0, np.zeros_like(y), y)
            y_1 = np.where(y < 0, np.zeros_like(y), np.ones_like(y))
            
            digit_y += np.histogram(x + stepsize, weights=y_0, range=(0,180), bins=int(math.ceil(180.0/stepsize)))[0]
            digit_norm += np.histogram(x + stepsize, weights=y_1, range=(0,180), bins=int(math.ceil(180.0/stepsize)))[0]
                
        nonzeros = np.nonzero(digit_norm)
        interp = interpolate.InterpolatedUnivariateSpline(bins[nonzeros], digit_y[nonzeros]/digit_norm[nonzeros])
        
        interpbins = np.arange(min(bins[nonzeros]), max(bins[nonzeros]), stepsize)
        interpbins = np.around(interpbins, decimals=3)
        interpy = interp(interpbins)
        
        outname = spec_path + spec_name + "_scan" + str(scan_num) + ".xye"
        outfile = open(outname, "w")
        for i in range(0, len(interpbins)):
            outfile.write(str(interpbins[i]) + "\t" + str(mult * interpy[i]) + "\t" + str(np.sqrt(mult * interpy[i])) + "\n")
        outfile.close()
    
        if plot_binning:
            plt.figure() #figsize=(20, 5))
            plt.plot(bins, digit_norm)
            #plt.savefig(outname[:-4] + ".pdf")

        if plot_integration:
            plt.figure() #figsize=(20, 5))
            plt.plot(interpbins, interpy)
            plt.xlabel('2$\\theta$ ($^\circ$)')
            plt.ylabel('Intensity (arb. units)')

        plt.show()
    return (interpbins, interpy)
        
def make_pixel_map(det_R, db_pixel):
    # Map each pixel into cartesian coordinates (x,y,z) in number of pixels from sample for direct beam conditions (2-theta = 0)
    # We only need to do this once, so we can be inefficient about it
    data = np.zeros((195, 487))
    tup = np.unravel_index(np.arange(len(data.flatten())), data.shape)
    xyz_map = np.vstack((tup[0], tup[1], det_R*np.ones(tup[0].shape)))
    xyz_map -= [[db_pixel[1]], 
                [db_pixel[0]], 
                [0]]
    return xyz_map