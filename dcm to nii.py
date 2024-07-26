# import SimpleITK as sitk
# import os
#
# Path = 'G:/AP/'  # DICOM存放的文件夹
# for i in range(218):
#     # 文件个数，我这里是100个
#     # "文件"+
#     # os.path.exists(path) 判断文件是否存在 固定语法，记住就行
#     # 定义一个变量判断文件是否存在,path指代路径,str(i)指代文件夹的名字
#     # name+str(i+1)为拼接名
#     # str(i+1)1，2，3，...
#     folderPath = os.path.join(Path + str(i + 1))
#     # print(folderPath)
#     # for folder in os.listdir(folderPath):
#     #     imagePath = os.path.join(folderPath, folder)  # dicom图片所在文件夹
#     #     # print(imagePath)
#     #     reader = sitk.ImageSeriesReader()
#     #
#     #     dicom_names = reader.GetGDCMSeriesFileNames(imagePath)
#     #     reader.SetFileNames(dicom_names)
#     #     image = reader.Execute()  # 获取到文件
#     #     sitk.WriteImage(image, folderPath + folder + '_image.nii.gz')
#     # print("完成")
#
# #coding=utf-8
# import SimpleITK as sitk
#
#
# def dcm2nii(dcms_path, nii_path):
# 	# 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
#     reader.SetFileNames(dicom_names)
#     image2 = reader.Execute()
# 	# 2.将整合后的数据转为array，并获取dicom文件基本信息
#     image_array = sitk.GetArrayFromImage(image2)  # z, y, x
#     origin = image2.GetOrigin()  # x, y, z
#     spacing = image2.GetSpacing()  # x, y, z
#     direction = image2.GetDirection()  # x, y, z
# 	# 3.将array转为img，并保存为.nii.gz
#     image3 = sitk.GetImageFromArray(image_array)
#     image3.SetSpacing(spacing)
#     image3.SetDirection(direction)
#     image3.SetOrigin(origin)
#     sitk.WriteImage(image3, nii_path)
#
#
# if __name__ == '__main__':
#     dcms_path = r'xxx\series1'  # dicom序列文件所在路径
#     nii_path = r'.\test.nii.gz'  # 所需.nii.gz文件保存路径
#     dcm2nii(dcms_path, nii_path)

#coding=utf-8
import SimpleITK as sitk
import os


def dcm2nii(dcms_path, nii_path):
	# 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
	# 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
	# 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)


if __name__ == '__main__':
    basepath=r'G:\DP'
    filenames = os.listdir(basepath)
    print(filenames)

    for name in filenames:
        dcms_path = os.path.join(basepath,name)  # dicom序列文件所在路径
        nii_path = os.path.join('G:/DP_nii',name+'.nii')  # 所需.nii.gz文件保存路径
        if dcms_path<r'G:\DP\16':
            continue
        try:
            dcm2nii(dcms_path, nii_path)
            print(dcms_path,"done")
        except:
            print(dcms_path,"gets wrong!!!")
