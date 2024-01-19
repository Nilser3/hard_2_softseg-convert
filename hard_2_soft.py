import argparse
import numpy as np
import nibabel as nib
import cv2
from scipy import ndimage
from scipy.signal import convolve2d
import os 

def main(args):
    mask_data_1 = nib.load(args.path_hard)
    header_mask_data_1 = mask_data_1.header
    mask_1 = np.array(mask_data_1.get_fdata())

    res_x = header_mask_data_1["pixdim"][2]
    res_y = header_mask_data_1["pixdim"][3]
    res_z = header_mask_data_1["pixdim"][4]

    path = os.path.join(args.path_output)
    os.mkdir(path)

    # Comand reslicing
    comand_1 = f"sct_resample -i {args.path_hard} -mm 0.1x0.1x{res_z} -o {args.path_output}/mask_rl.nii.gz"
    os.system(comand_1)

    mask_data_rl = nib.load( f"{args.path_output}/mask_rl.nii.gz")
    mask_rl = np.array(mask_data_rl.get_fdata())
    
    image_new= np.zeros((mask_rl.shape[0],mask_rl.shape[1],mask_rl.shape[2]))
    
    factor_a = 20
    factor_b = 10
    factor_c = 5
    factor_d = 2
    factor_e = 10 
    factor_f = 5
    factor_g = 0.25
    factor_h = 0.1
    factor_i = 1
    factor_j = 2
    factor_k = 0

    kernel = np.array([[factor_a , factor_b , factor_c , factor_d , factor_e , factor_f , factor_e , factor_d , factor_c , factor_b , factor_a],
		   [factor_b , factor_c , factor_d , factor_e , factor_f , factor_g , factor_f , factor_e , factor_d , factor_c , factor_b],
		   [factor_c , factor_d , factor_e , factor_f , factor_g , factor_h , factor_g , factor_f , factor_e , factor_d , factor_c],
		   [factor_d , factor_e , factor_f , factor_g , factor_h , factor_i , factor_h , factor_g , factor_f , factor_e , factor_d],
		   [factor_e , factor_f , factor_g , factor_h , factor_i , factor_j , factor_i , factor_h , factor_g , factor_f , factor_e],
		   [factor_f , factor_g , factor_h , factor_i , factor_j , factor_k , factor_j , factor_i , factor_h , factor_g , factor_f],
		   [factor_e , factor_f , factor_g , factor_h , factor_i , factor_j , factor_i , factor_h , factor_g , factor_f , factor_e],
		   [factor_d , factor_e , factor_f , factor_g , factor_h , factor_i , factor_h , factor_g , factor_f , factor_e , factor_d],
		   [factor_c , factor_d , factor_e , factor_f , factor_g , factor_h , factor_g , factor_f , factor_e , factor_d , factor_c],
		   [factor_b , factor_c , factor_d , factor_e , factor_f , factor_g , factor_f , factor_e , factor_d , factor_c , factor_b],
		   [factor_a , factor_b , factor_c , factor_d , factor_e , factor_f , factor_e , factor_d , factor_c , factor_b , factor_a]])
    
    for i in range (mask_rl.shape[2]):
        object_contour_eroded = ndimage.binary_erosion(mask_rl[:,:,i], iterations = 6)
        image_new[:,:,i] = convolve2d(object_contour_eroded, kernel, mode='same', boundary='wrap')
    image_new = cv2.normalize(image_new, None, 0,1, cv2.NORM_MINMAX)
    mask_rl_nii = nib.Nifti1Image(image_new, mask_data_rl.affine)
    nib.save(mask_rl_nii, f"{args.path_output}/mask_soft_pre.nii.gz")

    # Reslicing
    comand_2 = f"sct_resample -i {args.path_output}/mask_soft_pre.nii.gz -ref {args.path_hard} -o {args.path_output}/{args.path_output}_hard_2_soft.nii.gz"
    os.system(comand_2)
    
    # Remove tmp files
    comand_3 = f"rm {args.path_output}/mask_rl.nii.gz {args.path_output}/mask_soft_pre.nii.gz"
    os.system(comand_3)
    
    print("Soft mask saved at : ", f"{args.path_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    	description= 
    	'::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n'
    	'\n'
	    'Function to convert binary masks to soft masks. \n'
	    'Using: sct_resample \n'
        '\n'
    	'::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n',
        formatter_class=argparse.RawTextHelpFormatter )
    
    parser.add_argument("--path_hard", required=True, help="Path to binary mask (.nii.gz)")
    parser.add_argument("--path_output", required=True, help="Output folder")
    args = parser.parse_args()
    main(args)   
