import SimpleITK as sitk
def sampletransfer(image,mask):
    rif = sitk.ResampleImageFilter()
    rif.SetReferenceImage(image)
    rif.SetOutputPixelType(mask.GetPixelID())
    rif.SetInterpolator(sitk.sitkNearestNeighbor)
    resMask = rif.Execute(mask)
    return resMask
import os
basepath = 'G:/218_stan_radiomic/dp/'
folders = os.listdir(basepath)
print(folders)
for folder in folders:
    files = os.listdir(os.path.join(basepath,folder))
    print(*files)
    for file in files:
        if file.endswith("image.nrrd"):
            imageFile = os.path.join(basepath, folder, file)
        if file.endswith("label.nrrd"):
            maskFile = os.path.join(basepath, folder, file)
            #  print(imageFile, maskFile)
            _image = sitk.ReadImage(imageFile)
            _oldmask = sitk.ReadImage(maskFile)
            rightmask = sampletransfer(_image, _oldmask)
            newfile=file.replace('label.nrrd','new_label.nrrd')
            newmaskfile=os.path.join(basepath,folder,newfile)
            sitk.WriteImage(rightmask, newmaskfile)