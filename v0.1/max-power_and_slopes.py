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


# DESCRIPTION
# This code reads as input multiple short aperture squinted radargrams obtained
# using SOFA and generates two new products:
# - a new radargram calculated, for each sample, as the maximum power measured
#   on the same sample on all the input squinted radargrams
# - a map indicating the direction of maximum scattering related to each sample,
#   which is given by the squint angle of the input radargram corresponding to
#   the maximum power used for generating the previous output. This map is
#   created by hiding low power pixels (according to a user-defined PFA).
# File are supposed to be TIF files whose filenames include the
# corresponding squint angle.

import numpy as np
from utils import io


def read_img(filename):
    import gdal
    try:
        ds = gdal.Open(filename)
        b = ds.GetRasterBand(1)
        image = np.array(b.ReadAsArray())
        print("Image correctly read")
        print("size = "+str(image.shape))
    except:
        print("Reading problem")
        return -1
    return image
   



if __name__ == "__main__":
    
    MIN_ANGLE=-1.5  # degrees
    MAX_ANGLE=1.5
    ANGLE_STEP=0.05

    PFA=0.001
    N_BINS=2048
    
    noise_region_y_start=50
    noise_region_y_end=300
 
    INPUT_PREFIX = "./squinted_radargrams_folder/radargram_X_squint_"
    OUTPUT_PREFIX = "./output_"    
    SUFFIX=".tif"


# ------------ end of input parameters ------------

    n_squint_angles=int((MAX_ANGLE-MIN_ANGLE)/ANGLE_STEP+1)
    angles=np.arange(n_squint_angles)*ANGLE_STEP+MIN_ANGLE

    i_angle_abs=0
    ref_shape=0
    images_all=0
    BIN_RANGE_REV=np.arange(N_BINS-1,-1,-1)
    for i_angle in angles:
        print("id",i_angle_abs,"-->",i_angle)
        if np.abs(i_angle)>ANGLE_STEP*.5:
            str_angle=str(i_angle)
        else:    
            str_angle="0.0"
        filename=INPUT_PREFIX+str_angle+SUFFIX
        img=read_img(filename)

#   create stack of images according to size of the first read image
        if i_angle_abs==0:
            ref_shape=img.shape
            images_all=np.zeros([n_squint_angles,ref_shape[0],ref_shape[1]])

        images_all[i_angle_abs,:,:]=img

        i_angle_abs+=1






#        compute max image
    combined_max_power_image=images_all.max(axis=0)

    
#        compute a single noise threshold for the combined image
    noise_data_tmp=combined_max_power_image[noise_region_y_start:noise_region_y_end,:]
    h,b=np.histogram(noise_data_tmp,bins=N_BINS,density=True)                

    h*=(b[1]-b[0])

#    PFA threshold
    sum_h=0
    for ixx in BIN_RANGE_REV:
        sum_h+=h[ixx]
        if sum_h>=PFA:
            break
    ixx_thr=ixx
    noise_thr=b[ixx_thr]
    


#        compute argmax image to retrieve the direction (squint) of maximum scattering
#        which is directly related to the target slope
    dms_image=images_all.argmax(axis=0)
    dms_image=dms_image*ANGLE_STEP+MIN_ANGLE
    
#    threshold dms map
    dms_image[combined_max_power_image<noise_thr]=np.nan
    
    

    io.write_gtiff_f32(OUTPUT_PREFIX+"max-power"+SUFFIX,combined_max_power_image,debug_mode=True)    
    io.write_gtiff_f32(OUTPUT_PREFIX+"slope"+SUFFIX,dms_image,debug_mode=True)
