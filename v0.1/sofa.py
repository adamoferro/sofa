#!/usr/bin/python3

#   Copyright 2016-2018 Adamo Ferro
#
#   This file is part of SOFA.
#
#   SOFA is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   SOFA is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with SOFA. If not, see <http://www.gnu.org/licenses/>.
#
#   The use of SOFA or part of it for the creation of any sub-product
#   (e.g., scientific papers, posters, images, other softwares)
#   must be acknowledged.



import sys
import argparse
import math
import numpy as np
from scipy.signal import chirp
from scipy.ndimage.filters import gaussian_filter1d

from utils import io
from utils import geometry





def main(argv=None):
    
    if argv is None:
        argv=sys.argv


#   CONSTANTS
    cj=1j
    pi2=2.*math.pi

    c=299792458.            # propagation speed

#   SHARAD PARAMETERS
    FC=20e6                 # carrier frequency        
    F0=5e6                  # half bandwidth
    FAST_TIME_SAMPLING_FREQUENCY=80e6/3.
    DCG_DELAY=11.98e-6      # digital chirp generator delay, used in the calculation of the rxwin opening time

    reference_v_platform=3397.     # m/s TODO *** estimate from input data

    wc=pi2*FC
    lmbda_min=c/(FC+F0)    # Wavelength at highest frequency

    dt=1./FAST_TIME_SAMPLING_FREQUENCY

    n_input_samples=3600
    size_range_FFT=4096
    N_OUTPUT_SAMPLES=667*2
    


    SHARAD_CHIRP_DURATION=85.05e-6




    debug_mode=False

    parser=argparse.ArgumentParser(description='SHARAD Open Focusing Attempt (SOFA).')
    parser.add_argument('-ip','--input_path',required=True,help='path of input raw data.')
    parser.add_argument('-if','--input_filename_prefix',required=True,help='filename prefix of input raw data.')
    parser.add_argument('-op','--output_path',required=True,help='path of the output processed radargram.')
    parser.add_argument('-of','--output_filename_prefix',required=True,help='filename prefix of output processed radargram.')
    parser.add_argument('-fs','--frame_start',type=int,default='-1',help="first frame to be focused. Default is 'the first frame that can be focused depending on the focusing parameters'.")
    parser.add_argument('-fe','--frame_end',type=int,default='-1',help="last frame to be focused. Default is 'the last frame that can be focused depending on the focusing parameters'.")
    parser.add_argument('-fk','--frame_skip',type=int,default='26',help="spacing between frames to be focused. Default is 26.")
    parser.add_argument('-pri','--pulse_repetition_interval',type=float,default='1428',help="pulse repetition interval (us). Default is 1428.")
    parser.add_argument('-pf','--presumming_factor',type=int,default='4',help="raw data presumming factor. Default is 4.")
    parser.add_argument('-a','--half_aperture',type=int,default='768',help="synthetic aperture half-length expressed in number of frames. Default is 768.")
    parser.add_argument('-s','--squint_angle',type=float,default='0.',help="squint angle (degrees). Default is 0.")    
    parser.add_argument('-im','--input_mola_filename',required=True,help='MOLA data (MEGDR) filename.')
    parser.add_argument('-ims','--input_mola_sampling',type=float,default=1./128.,help="MOLA data sampling (degrees per pixel). Default is 1/128.")
    parser.add_argument('-imtM','--input_mola_lat_max',type=float,default='88.',help="MOLA data maximum latitude. Default is 88.")
    parser.add_argument('-imgm','--input_mola_long_min',type=float,default='0.',help="MOLA data minimum longitude. Default is 0.")
    parser.add_argument('-b','--block_size_max',type=int,default='5000',help="maximum size of frame blocks processed at the same time. Default is 5000.")
    parser.add_argument('-mb','--multilooking_doppler_bandwidth',type=float,default='0.8',help="Doppler bandwidth used for multilooking. Default is 0.8.")
    parser.add_argument('-db','--doppler_max_search_bandwidth',type=float,default='0.',help="Doppler bandwidth centered in 0 Hz for Doppler power max search. Default is 0.")
    parser.add_argument('-adw','--adjacent_doppler_windows',type=int,default='0',help="number of output adjacent Doppler windows. Default is 0.")
    parser.add_argument('-r','--range_migration',type=bool,default='True',help="correct range migration during focusing. Default is True.")
    parser.add_argument('-dc','--doppler_phase_compensation',type=bool,default='True',help="compensate Doppler phase during focusing. Default is True.")
    parser.add_argument('-rwf','--range_weighting_function',default='hanning',choices=['hanning','hamming','kaiser','none'],help="weighting function used in range weighting. Default is 'hann'.")
    parser.add_argument('-rwk','--range_weighting_kaiser_beta',type=float,default='2.',help="range Kaiser window beta parameter. Default is 2.")
    parser.add_argument('-awf','--azimuth_weighting_function',default='hanning',choices=['hanning','hamming','kaiser','none'],help="weighting function used in azimuth weighting. Default is 'hann'.")
    parser.add_argument('-awk','--azimuth_weighting_kaiser_beta',type=float,default='2.',help="azimuth Kaiser window beta parameter. Default is 2.")    
    parser.add_argument('-gn','--gauss_rec_n_filters',type=int,default='2',help="number of Gaussian filters used in Doppler peak reconstruction. Default is 2.")    
    parser.add_argument('-ga','--gauss_rec_alpha',type=float,default='0.66',help="alpha parameter used in Doppler peak reconstruction. Default is 0.66.")
    parser.add_argument('-p','--subprocesses',default='1',type=int,help="number of concurrent threads to be run (works only if the module multiprocessing is installed). Default is 1.")
    parser.add_argument('-d','--debug',action='store_true',default=False,help='debug mode. Default is not set.')
    parser.add_argument('-v','--version',action='version',version='%(prog)s 0.1')
    try:
        args=parser.parse_args()
    except:
        print("ERROR: problem while parsing input parameters.")
        exit()


    debug_mode=args.debug

#   TODO *** change variable style 
#   TODO *** check variable validity vs constraints 
    INPUT_DATA_FILENAME_BASE=args.input_path + "/" +args.input_filename_prefix 
    OUTPUT_PATH=args.output_path + "/"
    OUTPUT_FILENAME_PREFIX=args.output_filename_prefix
    PRI_acquisition=args.pulse_repetition_interval*1e-6
    presumming_factor=args.presumming_factor
    PRI_data=PRI_acquisition*presumming_factor
    
#    extra delay for the calculation of the rxwin opening time, which depends on the PRI range
    rxwin_PRI_delay=PRI_acquisition
    if PRI_acquisition>1500e-6:             # PRF > 670.24 Hz
        rxwin_PRI_delay=0

    
    
    MOLA_DATA_FILENAME=args.input_mola_filename
    MOLA_SAMPLING=args.input_mola_sampling      # degs per pixel
    MOLA_MAX_LAT=args.input_mola_lat_max        # top
    MOLA_MIN_LONG=args.input_mola_long_min      # left
    
    
    READ_BLOCK_SIZE_MAX_APPROX=args.block_size_max
    
#   TODO     *** calculate automatically if not given
    frames_to_be_focused_start=args.frame_start
    frames_to_be_focused_end=args.frame_end
    skip_frames=args.frame_skip
    
    n_half_aperture_frames=args.half_aperture
    n_aperture_frames=2*n_half_aperture_frames+1        # center sample is included and in the middle of aperture      



    block_overlapping_size=n_aperture_frames-skip_frames
    if block_overlapping_size<0:
        print("ERROR: frame skip greater than aperture size is not handled.")
        exit()


    coherence_time=(n_aperture_frames-1)*PRI_data
    du_global=PRI_data*reference_v_platform 
 
#    TODO 
    focusing_ML_doppler_half_bandwidth=args.multilooking_doppler_bandwidth/2.
    doppler_half_window_width=np.round(focusing_ML_doppler_half_bandwidth*coherence_time).astype("int")-1
#            avoid negative window widths
    doppler_half_window_width=max(0,doppler_half_window_width)
    
    ZERO_DOPPLER_SEARCH_BANDWIDTH=args.doppler_max_search_bandwidth/2.
    zero_doppler_search_half_window_width=np.round(ZERO_DOPPLER_SEARCH_BANDWIDTH*coherence_time).astype("int")-1
    zero_doppler_search_half_window_width=max(0,zero_doppler_search_half_window_width)

    if debug_mode:
        print("coherence time =",coherence_time)
        print("Doppler half window width =",doppler_half_window_width)
        print("ZDS half window width =",zero_doppler_search_half_window_width)




    N_HALF_ADJACENT_DOPPLER_WINDOWS=args.adjacent_doppler_windows


    CORRECT_RANGE_MIGRATION=args.range_migration    

    USE_DOPPLER_CORRECTION=args.doppler_phase_compensation
    

    
    
    N_GAUSSIAN_FILTERS=args.gauss_rec_n_filters
    GF_ALPHA=args.gauss_rec_alpha
    
    SQUINT_ANGLE_DEG=args.squint_angle

    SQUINT_ANGLE_RAD=SQUINT_ANGLE_DEG*math.pi/180.    # =0, look nadir; >0, look ahead; <0 look behind    


    
    n_subprocesses=args.subprocesses
    multiprocessing_enabled=False

    if n_subprocesses<1:
        n_subprocesses=1
    
    if n_subprocesses>1:
        try:
            from multiprocessing import Pool
            pool=Pool(n_subprocesses)
            exec_parallel_function=pool.starmap
            multiprocessing_enabled=True
            
        except:
            multiprocessing_enabled=False
            n_subprocesses=1
            print("WARNING: Multiprocessing not possible on this machine. Working with 1 process.")
    
    if multiprocessing_enabled==False:
        try:
            from itertools import starmap
            exec_parallel_function=starmap
        except:
            print("ERROR: itertools module not present.")
            exit()
    

    
    gf_power_coefficients=list()
    std_devs=list()
    if N_GAUSSIAN_FILTERS>0:
        gf_power_coefficients.append(GF_ALPHA)
        for i_gf in np.arange(N_GAUSSIAN_FILTERS-1):
            gf_power_coefficients.append((1.-np.sum(gf_power_coefficients))*GF_ALPHA)
        gf_power_coefficients.append(1-np.sum(gf_power_coefficients))

    #           GAUSSIAN STD DEVS
        std_dev_max=(np.round((focusing_ML_doppler_half_bandwidth*coherence_time)*10.))/10.
        std_devs=[(i_gf+1)*std_dev_max/N_GAUSSIAN_FILTERS for i_gf in np.arange(N_GAUSSIAN_FILTERS)]
     
        if debug_mode:
            print("Gaussian filters number:",N_GAUSSIAN_FILTERS)
            print("   power coefficients =",gf_power_coefficients,"(including no filter coeff.)")
            print("   std devs =",std_devs)
            
    else:
        if debug_mode:
            print("NO Gaussian filtering.")


            
            
            
#    TODO *** make input file format uniform
    input_rxwin_data_dt_pre=np.fromfile(INPUT_DATA_FILENAME_BASE+"_rxwin",dtype=np.float64)     # saved as number of samples spaced 37.5ns
    input_scx_data=np.fromfile(INPUT_DATA_FILENAME_BASE+"_x",dtype=np.float64,sep=" ")
    input_scy_data=np.fromfile(INPUT_DATA_FILENAME_BASE+"_y",dtype=np.float64,sep=" ")
    input_scz_data=np.fromfile(INPUT_DATA_FILENAME_BASE+"_z",dtype=np.float64,sep=" ")
    input_scvx_data=np.fromfile(INPUT_DATA_FILENAME_BASE+"_v_x",dtype=np.float64,sep=" ")
    input_scvy_data=np.fromfile(INPUT_DATA_FILENAME_BASE+"_v_y",dtype=np.float64,sep=" ")
    input_scvz_data=np.fromfile(INPUT_DATA_FILENAME_BASE+"_v_z",dtype=np.float64,sep=" ")
    input_lat_data=np.fromfile(INPUT_DATA_FILENAME_BASE+"_lat",dtype=np.float64,sep=" ")
    input_long_data=np.fromfile(INPUT_DATA_FILENAME_BASE+"_long",dtype=np.float64,sep=" ")
    input_long_data[input_long_data<0]=360+input_long_data[input_long_data<0]
    input_scr_data=np.sqrt(input_scx_data**2+input_scy_data**2+input_scz_data**2)
#    TODO *** SHARAD phase compensation not handled

    
    
#    Read input header file to retrieve raw data size
    input_data_file=io.envi_image(filename=INPUT_DATA_FILENAME_BASE,io_mode="read")
    input_data_file.read_header()
    n_input_frames=input_data_file.shape[0]


#    TODO *** specify the required MOLA data format    
    mola_data_file=io.envi_image(filename=MOLA_DATA_FILENAME,io_mode="read")
    if debug_mode:
        print("Reading MOLA data...")
    mola_data=mola_data_file.read()
    if debug_mode:
        print("MOLA data read")
    
    mola_x_vect,mola_y_vect=geometry.LL_to_MOLAXY(input_lat_data,input_long_data,MOLA_MAX_LAT,MOLA_MIN_LONG,MOLA_SAMPLING,mola_data.shape)
    
    mola_radius_profile=geometry.MOLAXY_to_radius_profile(mola_data,mola_x_vect,mola_y_vect)

   
    
    mola_data_file=None
    mola_x_vect=None
    mola_y_vect=None




#   Estimation of average number of frames corresponding to the defined squint offset
    input_sc_alt=input_scr_data-mola_radius_profile
    mean_altitude=np.average(input_sc_alt)
    squint_offset_approx_distance=-(mean_altitude*np.tan(SQUINT_ANGLE_RAD))*1.01    # TODO *** 1.01 because (abs) result is tipically underestimated
    squint_offset_approx_frames=int(np.round(squint_offset_approx_distance/du_global))
    
    if frames_to_be_focused_start==-1:
        frames_to_be_focused_start=np.max([n_half_aperture_frames-squint_offset_approx_frames+1,0])

    if frames_to_be_focused_end==-1:
        frames_to_be_focused_end=np.min([n_input_frames-n_half_aperture_frames-1-squint_offset_approx_frames,n_input_frames-1])

    if frames_to_be_focused_end<frames_to_be_focused_start:
        print("ERROR: last frame to be focused before first frame to be focused.")
        exit()


    if debug_mode:
        print("SC mean altitude =",mean_altitude)
        print("squint_offset_approx_distance =",squint_offset_approx_distance)
        print("squint_offset_approx_frames =",squint_offset_approx_frames)
        print("n_input_frames =",n_input_frames)


#   calculation of range of frames to be focused
    frames_to_be_focused=np.arange(frames_to_be_focused_start,frames_to_be_focused_end,skip_frames)
    n_frames_to_be_focused=len(frames_to_be_focused)
    if debug_mode:
        print("frames_to_be_focused_start =",frames_to_be_focused_start)
        print("frames_to_be_focused_end =",frames_to_be_focused_end)    
        print("n_frames_to_be_focused =",n_frames_to_be_focused)        


#   calculation of range of frames to be processed to focus the selected frames
    frames_to_be_processed_start=frames_to_be_focused_start-n_half_aperture_frames+squint_offset_approx_frames
    if frames_to_be_processed_start>frames_to_be_focused_start:
        first_block_start_offset=frames_to_be_processed_start-frames_to_be_focused_start
        frames_to_be_processed_start=frames_to_be_focused_start
        if debug_mode:
            print("\nNOTICE: first frame to be processed = first frame to be focused, as aperture is very delayed.")
    else:
        first_block_start_offset=0



    frames_to_be_processed_end=frames_to_be_focused_end+n_half_aperture_frames+squint_offset_approx_frames
    if frames_to_be_processed_end<frames_to_be_focused_end:
        frames_to_be_processed_end=frames_to_be_focused_end
        if debug_mode:
            print("\nNOTICE: last frame to be processed = last frame to be focused, as aperture is very anticipated.\n")

    frames_to_be_processed=np.arange(frames_to_be_processed_start,frames_to_be_processed_end,dtype="int")
    n_frames_to_be_processed=len(frames_to_be_processed)
    
    if debug_mode:
        print("n_frames_to_be_processed =",n_frames_to_be_processed)
        print("frames_to_be_processed_start =",frames_to_be_processed_start)
        print("frames_to_be_processed_end =",frames_to_be_processed_end)

    if frames_to_be_processed_start<0:
        print("ERROR: first frame to be processed is before the beginning of input raw data.")
        exit()

    if frames_to_be_processed_end>=input_data_file.shape[0]:
        print("ERROR: the last frame to be processed is after the end of input raw data.")
        exit()        


#   calculation of ids of frames to be focused wrt frames_to_be_processed
    frames_to_be_focused_within_frames_to_be_processed=frames_to_be_focused-frames_to_be_processed_start


#    select only data related to frames to be processed
    input_rxwin_data_dt_pre=input_rxwin_data_dt_pre[frames_to_be_processed]
    input_scx_data=input_scx_data[frames_to_be_processed]
    input_scy_data=input_scy_data[frames_to_be_processed]
    input_scz_data=input_scz_data[frames_to_be_processed]
    input_scvx_data=input_scvx_data[frames_to_be_processed]
    input_scvy_data=input_scvy_data[frames_to_be_processed]
    input_scvz_data=input_scvz_data[frames_to_be_processed]    
    input_lat_data=input_lat_data[frames_to_be_processed]
    input_long_data=input_long_data[frames_to_be_processed]    
    input_scr_data=input_scr_data[frames_to_be_processed]
    mola_radius_profile=mola_radius_profile[frames_to_be_processed]


# *** 1 ***
    

    input_theta=np.arctan2(input_scy_data,input_scx_data)
    input_phi=np.arccos(input_scz_data/input_scr_data)
    
    

#   shifts for final frame alignment
    input_scr_data_alignment=input_scr_data-geometry.get_mars_ellipsoid_radius(input_lat_data, input_long_data)
    input_scr_data_shifts_alignment=max(input_scr_data_alignment[frames_to_be_focused_within_frames_to_be_processed])-input_scr_data_alignment
    input_scr_data_shifts_alignment_dt_compr=(input_scr_data_shifts_alignment*2./c)/dt
    input_scr_data_shifts_alignment_t=(input_scr_data_shifts_alignment*2./c)


    input_rxwin_data_t=input_rxwin_data_dt_pre*dt+rxwin_PRI_delay-DCG_DELAY
    input_rxwin_data_dt_compr_shifts=input_rxwin_data_dt_pre-min(input_rxwin_data_dt_pre[frames_to_be_focused_within_frames_to_be_processed])
    input_rxwin_data_t_shifts=input_rxwin_data_t-min(input_rxwin_data_t[frames_to_be_focused_within_frames_to_be_processed])


     #   frame sample offsets for final frame alignment
    sample_start_shifts=np.round(input_rxwin_data_dt_compr_shifts+input_scr_data_shifts_alignment_dt_compr).astype("int")
    sample_start_shifts_max=max(sample_start_shifts[frames_to_be_focused_within_frames_to_be_processed])

    N_OUTPUT_SAMPLES+=sample_start_shifts_max       # update N_OUTPUT_SAMPLES
    OUTPUT_SAMPLES=np.arange(N_OUTPUT_SAMPLES)
    
    delay_start_shifts=input_rxwin_data_t_shifts+input_scr_data_shifts_alignment_t


    RANGE_WEIGHTING_FUNCTION=args.range_weighting_function
    RANGE_WEIGHTING_FUNCTION_KAISER_BETA=args.range_weighting_kaiser_beta
    
    range_weighting_window_original=1.
    RANGE_WEIGHTING_FUNCTION_STR=RANGE_WEIGHTING_FUNCTION
    if RANGE_WEIGHTING_FUNCTION=="none":
        RANGE_WEIGHTING_FUNCTION="ones"
    if RANGE_WEIGHTING_FUNCTION == "kaiser":
        range_weighting_window_original=np.kaiser(size_range_FFT/2,RANGE_WEIGHTING_FUNCTION_KAISER_BETA)
        RANGE_WEIGHTING_FUNCTION_STR+="-"+str(RANGE_WEIGHTING_FUNCTION_KAISER_BETA)
    else:
        range_weighting_window_original=eval("np."+RANGE_WEIGHTING_FUNCTION+"(size_range_FFT/2)")
        

    
    AZIMUTH_WEIGHTING_FUNCTION=args.azimuth_weighting_function
    AZIMUTH_WEIGHTING_FUNCTION_KAISER_BETA=args.azimuth_weighting_kaiser_beta
    
    azimuth_weighting_windows=1.
    AZIMUTH_WEIGHTING_FUNCTION_STR=AZIMUTH_WEIGHTING_FUNCTION
    if AZIMUTH_WEIGHTING_FUNCTION != "none":
        if AZIMUTH_WEIGHTING_FUNCTION == "kaiser":
            azimuth_weighting_window_original=np.kaiser(n_aperture_frames,AZIMUTH_WEIGHTING_FUNCTION_KAISER_BETA)
            AZIMUTH_WEIGHTING_FUNCTION_STR+="-"+str(AZIMUTH_WEIGHTING_FUNCTION_KAISER_BETA)
        else:
            azimuth_weighting_window_original=eval("np."+AZIMUTH_WEIGHTING_FUNCTION+"(n_aperture_frames)")
        azimuth_weighting_windows=azimuth_weighting_window_original*np.ones([N_OUTPUT_SAMPLES,1])



#     create complex phasor
    data_samples = np.arange(n_input_samples)
    phasor_samples = np.arange(size_range_FFT)

#    baseband conversion
    phasor_const=pi2 * (- FC) 
    phasor = np.exp(cj*(phasor_const * (phasor_samples / FAST_TIME_SAMPLING_FREQUENCY)))
    
# SYNTHETIC CHIRP
    t_chirp=np.arange(0,SHARAD_CHIRP_DURATION,1./FAST_TIME_SAMPLING_FREQUENCY)
    range_weighting_window=np.concatenate((np.zeros(int(size_range_FFT/4)),range_weighting_window_original,np.zeros(int(size_range_FFT/4))),0)
    
    f0=FC+F0
    t1=SHARAD_CHIRP_DURATION
    f1=FC-F0
    
    chirp2_real=chirp(t_chirp,f0,t1,f1)

    chirp2_complex=chirp2_real[:]
    chirp2_complex=chirp2_complex*np.exp(-1j*wc*t_chirp)
    chirp_fft_pre=(np.fft.fft(chirp2_complex,size_range_FFT))/np.sqrt(size_range_FFT)
    chirp_fft=np.concatenate((np.zeros(int(size_range_FFT/4),dtype="complex"),chirp_fft_pre[int(size_range_FFT/4):int(size_range_FFT/4)*3],np.zeros(int(size_range_FFT/4),dtype="complex")),0)

    
    chirp_fft=chirp_fft*range_weighting_window

    chirp_fft_conj=np.conj(chirp_fft).T
    
    t_frame_base=dt*OUTPUT_SAMPLES
    fftfreqs=np.fft.fftfreq(size_range_FFT,dt)
    




# *** 2 ***





# free memory
    mola_data=None

#        calculate block size for parallel processing
    read_block_size_optimal_approx=np.ceil(n_frames_to_be_processed/n_subprocesses)+block_overlapping_size
    if read_block_size_optimal_approx>READ_BLOCK_SIZE_MAX_APPROX:
        read_block_size_optimal_approx=READ_BLOCK_SIZE_MAX_APPROX
    n_frames_to_be_focused_within_block_max=int(np.ceil((read_block_size_optimal_approx-n_aperture_frames)/skip_frames+1))
    read_block_size=min(n_frames_to_be_processed,(n_frames_to_be_focused_within_block_max-1)*skip_frames+n_aperture_frames)
    read_min_block_size=n_aperture_frames
    n_blocks=int(np.ceil(n_frames_to_be_focused/n_frames_to_be_focused_within_block_max))
    id_last_block=n_blocks-1
    last_block_n_frames_to_be_focused_within_block=n_frames_to_be_focused-(n_blocks-1)*n_frames_to_be_focused_within_block_max
    last_block_n_frames_to_be_processed_within_block=int((last_block_n_frames_to_be_focused_within_block-1)*skip_frames+n_aperture_frames)
    
    if debug_mode:
        msg=""
        if read_block_size_optimal_approx==READ_BLOCK_SIZE_MAX_APPROX:
            msg="(MAXIMUM ALLOWED SIZE)"
        print("read_block_size_optimal_approx =",read_block_size_optimal_approx,msg)
        print("n_frames_to_be_focused_within_block_max =",n_frames_to_be_focused_within_block_max)
        print("read_block_size =",read_block_size)
        print("overlapping_size =",block_overlapping_size)
        print("read_min_block_size =",read_min_block_size)
        print("N BLOCKS =",n_blocks)
        print("last_block_n_frames_to_be_processed_within_block =",last_block_n_frames_to_be_processed_within_block)

#   setup file reader in block mode
    input_data_file.set_block_mode(read_block_size,block_overlapping_size,read_min_block_size)
    input_data_file.b_open()
    input_data_file.b_seek_row(frames_to_be_processed_start+first_block_start_offset)
    
#   initialize output images
#    output_range_compressed_image=np.zeros([N_OUTPUT_SAMPLES,n_frames_to_be_focused])
    output_image=np.zeros([N_OUTPUT_SAMPLES,n_frames_to_be_focused])                       
    output_doppler_adjacent_images=list()
    for i_odai in np.arange(N_HALF_ADJACENT_DOPPLER_WINDOWS*2):
        output_doppler_adjacent_images.append(np.zeros([N_OUTPUT_SAMPLES,n_frames_to_be_focused]))      #*******+sample_start_shifts_max
    
#    PROCESSING
    input_parameters=list()
    for block_id in np.arange(n_blocks):
        input_data, block_row_start=input_data_file.b_read_next_block()
        input_parameters.append((block_id,input_data,block_row_start,sample_start_shifts_max,n_frames_to_be_focused,id_last_block, last_block_n_frames_to_be_processed_within_block, frames_to_be_processed_start, size_range_FFT, data_samples, phasor, c,wc, input_rxwin_data_t, chirp_fft_conj, N_OUTPUT_SAMPLES, OUTPUT_SAMPLES, dt, input_scr_data, input_theta, input_phi, n_half_aperture_frames, n_aperture_frames, n_frames_to_be_focused_within_block_max, sample_start_shifts,CORRECT_RANGE_MIGRATION, input_scx_data, input_scy_data, input_scz_data, USE_DOPPLER_CORRECTION,lmbda_min, skip_frames, azimuth_weighting_windows, N_GAUSSIAN_FILTERS, std_devs, gf_power_coefficients, zero_doppler_search_half_window_width, N_HALF_ADJACENT_DOPPLER_WINDOWS, doppler_half_window_width,squint_offset_approx_frames,t_frame_base,mola_radius_profile,fftfreqs,delay_start_shifts))



#   Launch parallel processing after reading n_subprocesses blocks per time (save RAM)
        if ((block_id+1) % n_subprocesses==0) or block_id==n_blocks-1:
            results=exec_parallel_function(block_process,input_parameters)

            n_blocks_already_processed=int(block_id/n_subprocesses)*n_subprocesses
            for block_id2 in np.arange(block_id+1-n_blocks_already_processed):
                
                print("average real squint angle for block",block_id2+n_blocks_already_processed,"=",results[block_id2].avg_real_squint_angle*180./math.pi,"   mean aperture angle =",results[block_id2].avg_real_aperture_angle*180./math.pi)
                
#                TODO *** create sub-images with only needed frames, instead of whole image (to save memory)
#                output_range_compressed_image+=results[block_id2].output_range_compressed_image
                output_image+=results[block_id2].output_image
                for i_odai in np.arange(N_HALF_ADJACENT_DOPPLER_WINDOWS*2):
                    output_doppler_adjacent_images[i_odai]+=results[block_id2].output_doppler_adjacent_images[i_odai]
            
            results=None
            input_parameters=None
            input_parameters=list()
     

    io.write_gtiff_f32(OUTPUT_PATH+OUTPUT_FILENAME_PREFIX+"_"+str(frames_to_be_focused_start)+"-"+str(frames_to_be_focused_end)+"_skip-"+str(skip_frames)+"_ha-"+str(n_half_aperture_frames)+"_freq_ML-"+str(doppler_half_window_width*2+1)+"_"+str(focusing_ML_doppler_half_bandwidth)+"_ZDS-"+str(zero_doppler_search_half_window_width*2+1)+"_GAUSS-"+str(N_GAUSSIAN_FILTERS)+"_AZ-"+AZIMUTH_WEIGHTING_FUNCTION_STR+"_RG-"+RANGE_WEIGHTING_FUNCTION_STR+"_DPC-"+str(USE_DOPPLER_CORRECTION)+"_SQUINT_"+str(SQUINT_ANGLE_DEG)+".tif", (output_image),debug_mode) #[:,np.arange(0,n_frames_to_be_focused,15)])

    for i_odai in np.arange(N_HALF_ADJACENT_DOPPLER_WINDOWS*2):
        io.write_gtiff_f32(OUTPUT_PATH+OUTPUT_FILENAME_PREFIX+"_"+str(frames_to_be_focused_start)+"-"+str(frames_to_be_focused_end)+"_skip-"+str(skip_frames)+"_ha-"+str(n_half_aperture_frames)+"_freq_ML-"+str(doppler_half_window_width*2+1)+"_"+str(focusing_ML_doppler_half_bandwidth)+"_ZDS-"+str(zero_doppler_search_half_window_width*2+1)+"_GAUSS-"+str(N_GAUSSIAN_FILTERS)+"_AZ-"+AZIMUTH_WEIGHTING_FUNCTION_STR+"_RG-"+RANGE_WEIGHTING_FUNCTION_STR+"_DPC-"+str(USE_DOPPLER_CORRECTION)+"_SQUINT_"+str(SQUINT_ANGLE_DEG)+".tif", (output_doppler_adjacent_images[i_odai]),debug_mode) #[:,np.arange(0,n_frames_to_be_focused,15)])


# ----------- end of frame processing ------------
    
    
class block_outputs:
    def __init__(self,output_range_compressed_image,output_image,output_doppler_adjacent_images,avg_real_squint_angle,avg_real_aperture_angle):
        self.output_range_compressed_image=output_range_compressed_image
        self.output_image=output_image
        self.output_doppler_adjacent_images=output_doppler_adjacent_images
        self.avg_real_squint_angle=avg_real_squint_angle
        self.avg_real_aperture_angle=avg_real_aperture_angle
    
    def __del__(self):
        self.output_range_compressed_image=None
        self.output_image=None
        for i in np.arange(len(self.output_doppler_adjacent_images)):
            self.output_doppler_adjacent_images[i]=None
        self.output_doppler_adjacent_images=None
    
       




    

    
    
    
    
    
    
    
    
    
def block_process(block_id,input_data,block_row_start,sample_start_shifts_max,n_frames_to_be_focused,id_last_block, last_block_n_frames_to_be_processed_within_block, frames_to_be_processed_start, sizeFFT1, data_samples, phasor,c,wc, input_rxwin_data_t,chirp_fft_conj, N_OUTPUT_SAMPLES, OUTPUT_SAMPLES, dt_compr,input_scr_data, input_theta, input_phi, n_half_aperture_frames, n_aperture_frames, n_frames_to_be_focused_within_block_max, sample_start_shifts,CORRECT_RANGE_MIGRATION, input_scx_data, input_scy_data, input_scz_data, USE_DOPPLER_CORRECTION,lmbda_min, skip_frames, azimuth_weighting_windows, N_GAUSSIAN_FILTERS, std_devs, gf_power_coefficients, zero_doppler_search_half_window_width, N_HALF_ADJACENT_DOPPLER_WINDOWS, doppler_half_window_width,squint_offset_approx_frames,t_frame_base,mola_radius_profile,fftfreqs,delay_start_shifts):

    output_range_compressed_image=np.zeros([N_OUTPUT_SAMPLES,n_frames_to_be_focused])
    output_image=np.zeros([N_OUTPUT_SAMPLES,n_frames_to_be_focused])
    output_doppler_adjacent_images=list()
    for i_odai in np.arange(N_HALF_ADJACENT_DOPPLER_WINDOWS*2):
        output_doppler_adjacent_images.append(np.zeros([N_OUTPUT_SAMPLES,n_frames_to_be_focused]))

#    TODO ***
    input_data=input_data.T

#   last block may be smaller
    if block_id==id_last_block:
        input_data=input_data[:,np.arange(last_block_n_frames_to_be_processed_within_block)]


    read_block_size_real=input_data.shape[1]


    signals=np.zeros([sizeFFT1,read_block_size_real],dtype="complex")


#    TODO *** check if possible to convert in matrix calculations without killing RAM (as already happened...)
    for i_frame in np.arange(read_block_size_real):
        signals[data_samples,i_frame]=input_data[:,i_frame]*phasor[data_samples]
 

    input_data=None




    print("...matched filtering...")

    signals_fft=np.fft.fft(signals,axis=0)      # spectrum: 0-positive ... negative-0

#   FOR VERSION (slower and less RAM)   TODO *** try to convert in matrix format
    signals_fft_matched=np.zeros([len(chirp_fft_conj),read_block_size_real],dtype="complex")
    for i_frame in np.arange(read_block_size_real):
        signals_fft_matched[:,i_frame]=signals_fft[:,i_frame]*chirp_fft_conj

    signals=None
    signals_fft=None


    t_frame_base2=dt_compr*np.arange(sizeFFT1)
    phasor2=np.exp(1j*wc*t_frame_base2)    
    
    signals_matched=np.fft.ifft(signals_fft_matched,axis=0)
    for i_frame in np.arange(read_block_size_real):
        signals_matched[:,i_frame]*=phasor2
        
    signals_fft_matched=np.fft.fft(signals_matched,axis=0)
    




#   NOTE: frames to be focused could be OUTSIDE the block because of squint!!!
    frames_to_be_focused_within_block=np.arange(n_half_aperture_frames,read_block_size_real-n_half_aperture_frames,skip_frames)-squint_offset_approx_frames
    
    print("FOCUSING FROM ABS ",frames_to_be_focused_within_block[0]+block_row_start-frames_to_be_processed_start,"TO",frames_to_be_focused_within_block[-1]+block_row_start-frames_to_be_processed_start,"WITH SKIP",skip_frames)


    n_frames_to_be_focused_within_block=len(frames_to_be_focused_within_block)
    print("n_frames_to_be_focused_within_block =",n_frames_to_be_focused_within_block)



    i_frame_focusing=block_id*n_frames_to_be_focused_within_block_max
    angle_between_aperture_start_and_ref_sample=0
    angle_between_aperture_center_and_ref_sample=0
    angle_between_aperture_end_and_ref_sample=0
    for ref_frame in frames_to_be_focused_within_block:
        ref_frame_abs=block_row_start-frames_to_be_processed_start+ref_frame

#   aperture has final length = 2*n_half_aperture_frames+1
        aperture_range_offset=np.arange(squint_offset_approx_frames-n_half_aperture_frames,squint_offset_approx_frames+n_half_aperture_frames+1,dtype="int")
        aperture_frames_abs=ref_frame_abs+aperture_range_offset
        aperture_frames=ref_frame+aperture_range_offset


        if CORRECT_RANGE_MIGRATION:
            
#            range migration is corrected wrt MOLA radius: not necessary to correct for every single sample, error is negligible TODO *** check (already done, but investigate more)
            r_sample_ref=mola_radius_profile[ref_frame_abs]
            x_sample_ref=r_sample_ref*(np.cos(input_theta[ref_frame_abs]))*(np.sin(input_phi[ref_frame_abs]))
            y_sample_ref=r_sample_ref*(np.sin(input_theta[ref_frame_abs]))*(np.sin(input_phi[ref_frame_abs]))
            z_sample_ref=r_sample_ref*np.cos(input_phi[ref_frame_abs])
            
            sc_distance_to_ref_sample_within_aperture=np.sqrt((input_scx_data[aperture_frames_abs]-x_sample_ref)**2+(input_scy_data[aperture_frames_abs]-y_sample_ref)**2+(input_scz_data[aperture_frames_abs]-z_sample_ref)**2)
            sc_distance_to_ref_sample_within_aperture_ref=np.sqrt((input_scx_data[ref_frame_abs]-x_sample_ref)**2+(input_scy_data[ref_frame_abs]-y_sample_ref)**2+(input_scz_data[ref_frame_abs]-z_sample_ref)**2)

#           t of beginning of frames wrt tx -rxwin
            t_within_frames_within_aperture=sc_distance_to_ref_sample_within_aperture*2./c-input_rxwin_data_t[aperture_frames_abs]
            t_within_frames_within_aperture_ref=sc_distance_to_ref_sample_within_aperture_ref*2./c-input_rxwin_data_t[ref_frame_abs]


            sc_position_wrt_ref_sample=(input_scx_data[ref_frame_abs]-x_sample_ref,input_scy_data[ref_frame_abs]-y_sample_ref,input_scz_data[ref_frame_abs]-z_sample_ref)
            sc_aperture_start_position_wrt_ref_sample=(input_scx_data[ref_frame_abs+squint_offset_approx_frames-n_half_aperture_frames]-x_sample_ref,input_scy_data[ref_frame_abs+squint_offset_approx_frames-n_half_aperture_frames]-y_sample_ref,input_scz_data[ref_frame_abs+squint_offset_approx_frames-n_half_aperture_frames]-z_sample_ref)
            sc_aperture_center_position_wrt_ref_sample=(input_scx_data[ref_frame_abs+squint_offset_approx_frames]-x_sample_ref,input_scy_data[ref_frame_abs+squint_offset_approx_frames]-y_sample_ref,input_scz_data[ref_frame_abs+squint_offset_approx_frames]-z_sample_ref)
            sc_aperture_end_position_wrt_ref_sample=(input_scx_data[ref_frame_abs+squint_offset_approx_frames+n_half_aperture_frames]-x_sample_ref,input_scy_data[ref_frame_abs+squint_offset_approx_frames+n_half_aperture_frames]-y_sample_ref,input_scz_data[ref_frame_abs+squint_offset_approx_frames+n_half_aperture_frames]-z_sample_ref)
            
            angle_between_aperture_start_and_ref_sample+=geometry.angle_between(sc_position_wrt_ref_sample,sc_aperture_start_position_wrt_ref_sample)
            angle_between_aperture_center_and_ref_sample+=geometry.angle_between(sc_position_wrt_ref_sample,sc_aperture_center_position_wrt_ref_sample)
            angle_between_aperture_end_and_ref_sample+=geometry.angle_between(sc_position_wrt_ref_sample,sc_aperture_end_position_wrt_ref_sample)
            
            t_sample_compr_ref=(t_frame_base.T+input_rxwin_data_t[ref_frame_abs])
            r_sample_ref=np.array([input_scr_data[ref_frame_abs]-t_sample_compr_ref*c/2.]).T
            x_sample_ref=r_sample_ref*(np.cos(input_theta[ref_frame_abs]))*(np.sin(input_phi[ref_frame_abs]))
            y_sample_ref=r_sample_ref*(np.sin(input_theta[ref_frame_abs]))*(np.sin(input_phi[ref_frame_abs]))
            z_sample_ref=r_sample_ref*np.cos(input_phi[ref_frame_abs])


            sc_distance_to_ref_sample_within_aperture_matrix=np.sqrt((input_scx_data[aperture_frames_abs]*np.ones([N_OUTPUT_SAMPLES,1])-x_sample_ref*np.ones([1,n_aperture_frames]))**2+(input_scy_data[aperture_frames_abs]*np.ones([N_OUTPUT_SAMPLES,1])-y_sample_ref*np.ones([1,n_aperture_frames]))**2+(input_scz_data[aperture_frames_abs]*np.ones([N_OUTPUT_SAMPLES,1])-z_sample_ref*np.ones([1,n_aperture_frames]))**2)

            delays_within_aperture=-(t_within_frames_within_aperture-t_within_frames_within_aperture_ref)


            if USE_DOPPLER_CORRECTION:
    #               phase correction "US style", only referred to a reference sample
                doppler_phase_corr_within_aperture=-4.*math.pi*sc_distance_to_ref_sample_within_aperture_matrix/lmbda_min
                                
            else:
                doppler_phase_corr_within_aperture=np.ones([N_OUTPUT_SAMPLES,n_aperture_frames])
        else:
            delays_within_aperture=sample_start_shifts[aperture_frames_abs]*dt_compr
            doppler_phase_corr_within_aperture=np.ones([N_OUTPUT_SAMPLES,n_aperture_frames])

        signals_matched_within_aperture=np.zeros([N_OUTPUT_SAMPLES,n_aperture_frames],dtype="complex")
               
        for i_aperture_frame in aperture_frames-aperture_frames[0]:
            delay_vector_fft=np.exp(-1j*2.*math.pi*fftfreqs*(delays_within_aperture[i_aperture_frame]+delay_start_shifts[ref_frame_abs]))
            signals_matched_within_aperture[:,i_aperture_frame]=np.fft.ifft(signals_fft_matched[:,i_aperture_frame+aperture_frames[0]]*delay_vector_fft)[OUTPUT_SAMPLES]


#                Doppler correction on whole aperture
        signals_matched_within_aperture*=np.exp(1j*doppler_phase_corr_within_aperture)
        signals_matched_within_aperture*=azimuth_weighting_windows


        signals_matched_within_aperture_azimuth_fft=np.fft.fftshift(np.fft.fft(signals_matched_within_aperture,axis=1),axes=(1,))

        signals_matched_within_aperture_azimuth_fft=np.abs(signals_matched_within_aperture_azimuth_fft)**2.




        if N_GAUSSIAN_FILTERS>0:
            signals_matched_within_aperture_upsampled_azimuth_fft_g=list()

            for std_dev_tmp in std_devs:
                signals_matched_within_aperture_upsampled_azimuth_fft_g.append(gaussian_filter1d(signals_matched_within_aperture_azimuth_fft,std_dev_tmp,axis=1))

            signals_matched_within_aperture_azimuth_fft*=gf_power_coefficients[0]
            for i_std_dev in np.arange(N_GAUSSIAN_FILTERS):
                signals_matched_within_aperture_azimuth_fft+=gf_power_coefficients[N_GAUSSIAN_FILTERS-1-i_std_dev]*signals_matched_within_aperture_upsampled_azimuth_fft_g[i_std_dev]

        frame_power_max_index=(n_half_aperture_frames-zero_doppler_search_half_window_width)+np.argmax(signals_matched_within_aperture_azimuth_fft[:,n_half_aperture_frames-zero_doppler_search_half_window_width:n_half_aperture_frames+zero_doppler_search_half_window_width+1],axis=1)
        frame_multilooking_start=frame_power_max_index-doppler_half_window_width
        output_frame_power=np.zeros(N_OUTPUT_SAMPLES)
        for i_output_sample in OUTPUT_SAMPLES:
            output_frame_power[i_output_sample]=np.sum(signals_matched_within_aperture_azimuth_fft[i_output_sample,frame_multilooking_start[i_output_sample]:frame_multilooking_start[i_output_sample]+doppler_half_window_width*2+1])/(doppler_half_window_width*2+1)

        output_frame_power_odai=list()
        for i_odai in np.arange(N_HALF_ADJACENT_DOPPLER_WINDOWS):

#                        negative wrt zero
            frame_power_max_index_odai=(-(i_odai)-(i_odai+1)*2*zero_doppler_search_half_window_width+n_half_aperture_frames-zero_doppler_search_half_window_width-1)+np.argmax(signals_matched_within_aperture_azimuth_fft[:,-(i_odai)-(i_odai+1)*2*zero_doppler_search_half_window_width+n_half_aperture_frames-zero_doppler_search_half_window_width-1:-(i_odai)-(i_odai+1)*2*zero_doppler_search_half_window_width+n_half_aperture_frames-zero_doppler_search_half_window_width+2*zero_doppler_search_half_window_width],axis=1)
            frame_multilooking_start_odai=frame_power_max_index_odai-doppler_half_window_width
            output_frame_power_odai.append(np.zeros(N_OUTPUT_SAMPLES))
            for i_output_sample in OUTPUT_SAMPLES:
                output_frame_power_odai[i_odai*2][i_output_sample]=np.sum(signals_matched_within_aperture_azimuth_fft[i_output_sample,frame_multilooking_start_odai[i_output_sample]:frame_multilooking_start_odai[i_output_sample]+doppler_half_window_width*2+1])/(doppler_half_window_width*2+1)

#                        positive wrt zero
            frame_power_max_index_odai=(+(i_odai)+(i_odai+1)*2*zero_doppler_search_half_window_width+n_half_aperture_frames-zero_doppler_search_half_window_width+1)+np.argmax(signals_matched_within_aperture_azimuth_fft[:,+(i_odai)+(i_odai+1)*2*zero_doppler_search_half_window_width+n_half_aperture_frames-zero_doppler_search_half_window_width+1:+(i_odai)+(i_odai+1)*2*zero_doppler_search_half_window_width+n_half_aperture_frames-zero_doppler_search_half_window_width+2*zero_doppler_search_half_window_width+2],axis=1)
            frame_multilooking_start_odai=frame_power_max_index_odai-doppler_half_window_width
            output_frame_power_odai.append(np.zeros(N_OUTPUT_SAMPLES))
            for i_output_sample in OUTPUT_SAMPLES:
                output_frame_power_odai[i_odai*2+1][i_output_sample]=np.sum(signals_matched_within_aperture_azimuth_fft[i_output_sample,frame_multilooking_start_odai[i_output_sample]:frame_multilooking_start_odai[i_output_sample]+doppler_half_window_width*2+1])/(doppler_half_window_width*2+1)


        output_image[OUTPUT_SAMPLES,i_frame_focusing]=output_frame_power[:]


        for i_odai in np.arange(N_HALF_ADJACENT_DOPPLER_WINDOWS*2):
            output_doppler_adjacent_images[i_odai][sample_start_shifts[ref_frame_abs]:sample_start_shifts[ref_frame_abs]+N_OUTPUT_SAMPLES,i_frame_focusing]=output_frame_power_odai[i_odai][:]

        i_frame_focusing+=1
    
    angle_between_aperture_start_and_ref_sample/=n_frames_to_be_focused_within_block
    angle_between_aperture_center_and_ref_sample/=n_frames_to_be_focused_within_block
    angle_between_aperture_end_and_ref_sample/=n_frames_to_be_focused_within_block
    
    if abs(squint_offset_approx_frames)<n_half_aperture_frames:
        angle_between_aperture_start_and_ref_sample*=-1
    aperture_angle=np.abs(angle_between_aperture_end_and_ref_sample-angle_between_aperture_start_and_ref_sample)
    
    
    return block_outputs(output_range_compressed_image,output_image,output_doppler_adjacent_images,angle_between_aperture_center_and_ref_sample,aperture_angle)
    
    
    
    

if __name__ == "__main__":
    sys.exit(main())
    
