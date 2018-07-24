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


def write_gtiff_f32(filename,img_np_array,debug_mode=False):
    from osgeo import gdal
    try:
        driver = gdal.GetDriverByName('GTiff')

        ds = driver.Create(filename,img_np_array.shape[1],img_np_array.shape[0],1,gdal.GDT_Float32)

        ds.SetGeoTransform((
            0,                      # 0: x_min
            1,                      # 1: pixel_size_x
            0,                      # 2: 0
            img_np_array.shape[0],  # 3: y_max
            0,                      # 4: 0
            -1))                    # 5: -pixel_size_y

        ds.GetRasterBand(1).WriteArray(img_np_array)
        ds.FlushCache()             # write to disk.
        if debug_mode:
            print("Image written to",filename)
    except:
        print("ERROR: problem writing image to disk.")
        return -1
    return 0
 
def write_gtiff_f32_multiband(filename,img_np_array,debug_mode=False):
    from osgeo import gdal
    try:
        driver = gdal.GetDriverByName('GTiff')

        n_bands=img_np_array.shape[0]
        ds = driver.Create(filename,img_np_array.shape[2],img_np_array.shape[1],n_bands,gdal.GDT_Float32)

        ds.SetGeoTransform((
            0,                      # 0: x_min
            1,                      # 1: pixel_size_x
            0,                      # 2: 0
            img_np_array.shape[1],  # 3: y_max
            0,                      # 4: 0
            -1))                    # 5: -pixel_size_y

        for i_band in np.arange(1,n_bands+1,dtype="int"):
            ds.GetRasterBand(int(i_band)).WriteArray(img_np_array[i_band-1,:,:])
        ds.FlushCache()             # write to disk.
        if debug_mode:
            print("Image written to",filename)
    except:
        print("ERROR: problem writing image to disk.")
        return -1
    return 0 
 
import numpy as np

class envi_image:
    """
    TODO *******************
    """

    def __init__(self,filename="",io_mode="read"):
        
        self._set_filenames(filename)
        self._reset()
        self.io_mode=io_mode
        self._check_mode()
        
        self.b_is_open=False
        self.b_file_pointer=None
        self._b_reset()
    
    def _set_filenames(self,filename):
        self.filename=filename
        self.header_filename=filename+".hdr"
        self.header_filename2=filename+".HDR"
        
    
    def _reset(self):
        self.shape=None
        self.envi_data_type=None     # 4 = float32, 5 = float64
        self.data_type=None


    def _b_reset(self):
        # used in block mode
        if self.b_is_open:
            self.b_file_pointer.close()
        self.b_mode=False
        self.b_read_n_rows=0
        self.b_read_n_rows_overlapping=0
        self.b_read_min_n_rows=0
        self.b_n_read_rows=0
        self.b_file_pointer=None
        self.b_is_open=False
        self.b_row_position=0
        
        

    def _check_mode(self):
        if self.io_mode!="read" and self.io_mode!="write":
            raise ValueError("ERROR: parameter mode could be only \"read\" or \"write\".")


    def set_block_mode(self,b_read_n_rows,b_read_n_rows_overlapping,b_read_min_n_rows):
        if b_read_n_rows>0 and b_read_n_rows_overlapping>=0 and b_read_n_rows>b_read_n_rows_overlapping:
            self.b_mode=True
            self.b_read_n_rows=b_read_n_rows
            self.b_read_n_rows_overlapping=b_read_n_rows_overlapping
            self.b_read_min_n_rows=b_read_min_n_rows
        else:
            raise ValueError("ERROR: not consistent block read and overlapping sizes.")
        

    def set_envi_data_type(self,envi_data_type):
        if envi_data_type==2:
            self.envi_data_type=2
            self.data_type=np.dtype(np.int16)
        elif envi_data_type==4:
            self.envi_data_type=4
            self.data_type=np.dtype(np.float32)
        elif envi_data_type==5:
            self.envi_data_type=5
            self.data_type=np.dtype(np.float64)
        else:
            raise ValueError("ERROR: not supported ENVI data type.")
        

    def set_filename(self,filename):
        self._set_filenames(filename)
        self._reset()
        self._b_reset()
        
    def set_mode(self,mode):
        self.io_mode=mode
        self._check_mode()
        
#        setting the mode triggers the reset of the class
        self._reset()
        self._b_reset()
        
        

    def read_header(self):
        if self.io_mode=="read":
            if self.filename=="":
                raise ValueError("ERROR: filename not set.")

            is_header_read=False
            try:
                with open(self.header_filename,"rt") as hfp:
                    header=hfp.read().splitlines()
                    is_header_read=True
            except EnvironmentError:
                try:
                    with open(self.header_filename2,"rt") as hfp:
                        header=hfp.read().splitlines()
                        is_header_read=True
                except EnvironmentError:
                    print("ERROR: problem while opening header file.")

            if is_header_read:
                size_x=[int(s[s.find("=")+1:].strip()) for s in header if "samples" in s][0]
                size_y=[int(s[s.find("=")+1:].strip()) for s in header if "lines" in s][0]
                envi_data_type=[int(s[s.find("=")+1:].strip()) for s in header if "data type" in s][0]
                
                if size_x>0 and size_y>0 and (envi_data_type==4 or envi_data_type==5 or envi_data_type==2):
                    self.shape=(size_y,size_x)
                    self.set_envi_data_type(envi_data_type)
#                    print(self.shape)
#                    print(self.data_type)                    
                    return True
                else:
                    print("WARNING: header does not contain valid data or data type is not supported. No data will be read.")
                    return False

        else:
            print("WARNING: invalid call to method read_header, mode is write.")


    def b_seek_row(self,n_rows):
        if self.io_mode=="read" and self.b_mode==True and self.b_is_open==True:
            if self.shape is None:
                self.read_header()
            
            if self.shape is not None:
                try:
                    self.b_file_pointer.seek(int(n_rows)*self.shape[1]*self.data_type.itemsize)
                    self.b_row_position=n_rows
                except EnvironmentError:
                    print("ERROR: problem while doing seek in input data file. Check seek offset.")
                
        elif self.b_mode==False:
            print("WARNING: invalid call to method b_seek_row, block mode is not enabled.")
        elif self.b_is_open==False:
            print("WARNING: invalid call to method b_seek_row, data file has not been open.")
        else:
            print("WARNING: invalid call to method b_seek_row, mode is write.")
                

    def read(self):
        if self.io_mode=="read" and self.b_mode==False:
            if self.shape is None:
                self.read_header()

            if self.shape is not None:
                try:
                    image=np.fromfile(self.filename,dtype=self.data_type,count=self.shape[0]*self.shape[1]).reshape(self.shape)
                except EnvironmentError:
                    print("ERROR: problem while reading data file.")
                else:
                    return image

        elif self.b_mode==True:
            print("WARNING: invalid call to method read, block mode is enabled.")
        else:
            print("WARNING: invalid call to method read, mode is write.")


    def b_open(self):
        if self.b_mode==True:
            if self.filename=="":
                raise ValueError("ERROR: filename not set.")        
        
            if self.b_is_open==True:
                print("WARNING: data file is already open.")
            else:
                try:
                    if self.io_mode=="read":
                        if self.shape is None:
                            self.read_header()

                        if self.shape is not None:                    
                            self.b_file_pointer=open(self.filename,"rb")
#                            print("file aperto in rb")
                    else:
    #                    TODO ****** da completare
                        self.b_file_pointer=open(self.filename,"wb")
                except EnvironmentError:
                    print("ERROR: problem while opening data file.")
                else:
                    self.b_is_open=True
        else:
            print("WARNING: invalid call to method b_open, block mode is not enabled.")


    def b_read_next_block(self):
        if self.io_mode=="read" and self.b_mode==True and self.b_is_open==True:
#            print("BEFORE READ",self.b_file_pointer.tell())
#            print(self.b_read_n_rows)
            try:
                block_row_start=self.b_row_position
                block=np.fromfile(self.b_file_pointer,dtype=self.data_type,count=self.shape[1]*self.b_read_n_rows).reshape((-1,self.shape[1]))
#                print("AFTER READ",self.b_file_pointer.tell())
                
#                if block.shape[0]<self.b_read_n_rows:
#                    print("EOF")

                if block.shape[0]<self.b_read_min_n_rows:
                    block=np.array([])
                else:
                    self.b_file_pointer.seek(-self.b_read_n_rows_overlapping*self.shape[1]*self.data_type.itemsize,1)
                    self.b_row_position+=self.b_read_n_rows-self.b_read_n_rows_overlapping
#                    print("AFTER SEEK",self.b_file_pointer.tell())
                
                return block, block_row_start

            except EnvironmentError:
                print("ERROR: problem while reading input data.")
                return None
            
            
        elif self.b_mode==False:
            print("WARNING: invalid call to method b_read_next_block, block mode is not enabled.")
        elif self.b_is_open==False:
            print("WARNING: invalid call to method b_read_next_block, data file has not been open.")
        else:
            print("WARNING: invalid call to method b_read_next_block, mode is write.")
                
                
    def b_close(self):
        if self.b_is_open:
            self.b_file_pointer.close()
            self.b_is_open=False


    def __del__(self):
        if (self.b_file_pointer is not None) and (not self.b_file_pointer.closed):
            self.b_file_pointer.close()
