import SimpleITK as sitk
import os
# import radiomics
# from radiomics import featureextractor
import pandas as pd
import numpy as np
from scipy import ndimage
import pandas as pd



# imagePath是图像nrrd,preference_path是参考nrrd的路径
# 定义原始图像和ROI文件
basepath ='G:/218_stan_radiomic/test/'
folders = os.listdir(basepath)
print(folders)
for folder in folders:
    files = os.listdir(os.path.join(basepath,folder))
    print(*files)
    for file in files:
        if file.endswith("image.nrrd"):
            img_path = os.path.join(basepath, folder, file)
        if file.endswith("new_label.nrrd"):
            roi_path = os.path.join(basepath, folder, file)


# img_path = r'original.nii'
# roi_path = r'mask.nii'


tumorImage = sitk.ReadImage(img_path)

# 设置重采样，保持层间距不变，进行重采样以减小工作量
newSpacing=[1.0,1.0,tumorImage.GetSpacing()[2]]
def resample_image_itk (img_path,preference_path=img_path,newSpacing = newSpacing, resamplemethod=sitk.sitkLinear):

# 进行resample
 Niimage_resample = resample_image_itk(img_path,preference_path=img_path,newSpacing = newSpacing, resamplemethod=sitk.sitkLinear) # resample #这里要注意：mask用最近邻插值sitk.sitkNearestNeighbor，CT图像用线性插值
# Resample之后的ROI只能和resample之后的原图一起提取组学特征，spacing不匹配的话就会报错，所以原图也要Resample
 image_array = sitk.GetArrayFromImage(Niimage_resample)


mask_img_resample = resample_image_itk(roi_path,preference_path=img_path,newSpacing = newSpacing, resamplemethod=sitk.sitkNearestNeighbor)
# '进行拓展'
mask_img_arr = sitk.GetArrayFromImage(mask_img_resample)
iteration = 3 # newSpacing是1mm,扩展3mm
mask_img_arr_expand = expand_dilation(mask_img_arr,iterations=iteration)
mask_img_arr_border = diffROI(mask_img_arr,mask_img_arr_expand)
outNII_path = '{}_resample.nrrd'.format('3mm_origin')
array2nii(image_array,outNII_path,Niimage_resample)

outROI_path = 'Ex{}.nrrd'.format('_3mm')
array2nii(mask_img_arr_border,outROI_path,Niimage_resample)
def resample_image_itk(imagePath, preferenceNii, newSpacing=[3.0, 3.0, 3.0],
                       resamplemethod=sitk.sitkLinear):  # 这里要注意：mask用最近邻插值，CT图像用线性插值，sitk.sitkNearestNeighbor
    image = sitk.ReadImage(preferenceNii)

    resample = sitk.ResampleImageFilter()

    resample.SetInterpolator(resamplemethod)  # 这里要注意：mask用最近邻插值，CT图像用线性插值 ##
    resample.SetOutputDirection(image.GetDirection())  ##
    resample.SetOutputOrigin(image.GetOrigin())  ##

    newSpacing = np.array(newSpacing, float)
    newSize = image.GetSize() / newSpacing * image.GetSpacing()
    newSize = np.around(newSize, decimals=0)
    newSize = newSize.astype(np.int)

    resample.SetSize(newSize.tolist())  ##
    resample.SetOutputSpacing(newSpacing)  ##
    image_resample = sitk.ReadImage(imagePath)
    newimage = resample.Execute(image_resample)
    return newimage  # 返回sitk类型的数据


def expand_dilation(mask_img_arr, iterations=5):
    shape_nrrd = mask_img_arr.shape
    mask_img_arr_expand = np.zeros(shape_nrrd)
    for index in range(shape_nrrd[0]):
        mask_img_arr_expand[index, :, :] = scipy.ndimage.binary_dilation(mask_img_arr[index, :, :],
                                                                         iterations=iterations).astype('uint16')
    return mask_img_arr_expand


def diffROI(mask_img_arr, mask_img_arr_expand):
    # 通过传入原始ROI mask_img_arr和扩展了5mm的ROI mask_img_arr_expand
    # 返回扩展的5mm的ROI区域
    # 首先把二值ROI都转为0和1
    mask_img_arr[mask_img_arr != 0] = 1
    mask_img_arr_expand[mask_img_arr_expand != 0] = 1

    shape_nrrd = mask_img_arr.shape
    mask_img_arr_border = np.zeros(shape_nrrd)
    for index in range(shape_nrrd[0]):
        for x in range(shape_nrrd[1]):
            for y in range(shape_nrrd[2]):
                if mask_img_arr[index, x, y] != mask_img_arr_expand[index, x, y]:
                    mask_img_arr_border[index, x, y] = 1
    return mask_img_arr_border


def array2nii(image_array, out_path, NIIimage_resample):
    ## image_array是矩阵，out_path是带文件名的路径，NIIimage_resample是sitk_obj
    # 1.构建nrrd阅读器
    image2 = NIIimage_resample
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, out_path)