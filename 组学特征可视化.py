# -*- coding: utf-8 -*-
import logging
import os

import SimpleITK as sitk
import six
import radiomics
# import pyradiomics

# from radiomics import featureextractor, getFeatureClasses
from radiomics import featureextractor

# Example usage
extractor = featureextractor.RadiomicsFeatureExtractor()
# exit()
#2
def tqdmProgressbar():
  """
  显示进度条功能
  This function will setup the progress bar exposed by the 'tqdm' package.
  Progress reporting is only used in PyRadiomics for the calculation of GLCM and GLSZM in full python mode, therefore
  enable GLCM and full-python mode to show the progress bar functionality

  N.B. This function will only work if the 'click' package is installed (not included in the PyRadiomics requirements)
  """
  global extractor

  radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar

  import tqdm
  radiomics.progressReporter = tqdm.tqdm
#3
def clickProgressbar():
  """
  This function will setup the progress bar exposed by the 'click' package.
  Progress reporting is only used in PyRadiomics for the calculation of GLCM and GLSZM in full python mode, therefore
  enable GLCM and full-python mode to show the progress bar functionality.

  Because the signature used to instantiate a click progress bar is different from what PyRadiomics expects, we need to
  write a simple wrapper class to enable use of a click progress bar. In this case we only need to change the 'desc'
  keyword argument to a 'label' keyword argument.

  N.B. This function will only work if the 'click' package is installed (not included in the PyRadiomics requirements)
  """
  global extractor

  # Enable the GLCM class to show the progress bar
  extractor.enableFeatureClassByName('glcm')

  radiomics.setVerbosity(logging.INFO)  # Verbosity must be at least INFO to enable progress bar

  import click

  class progressWrapper:
    def __init__(self, iterable, desc=''):
      # For a click progressbar, the description must be provided in the 'label' keyword argument.
      self.bar = click.progressbar(iterable, label=desc)

    def __iter__(self):
      return self.bar.__iter__()  # Redirect to the __iter__ function of the click progressbar

    def __enter__(self):
      return self.bar.__enter__()  # Redirect to the __enter__ function of the click progressbar

    def __exit__(self, exc_type, exc_value, tb):
      return self.bar.__exit__(exc_type, exc_value, tb)  # Redirect to the __exit__ function of the click progressbar

  radiomics.progressReporter = progressWrapper

  #4
  testCase = 'lung2'
  imageName, maskName = radiomics.getTestCase(testCase, r"D:\GE影像组学特征可视化\影像组学可视化20231219\影像组学可视化\lung2")
  paramsFile = os.path.abspath(r"D:\GE影像组学特征可视化\影像组学可视化20231219\影像组学可视化\\exampleVoxel.yaml")

  # imageName = r"C:\\Users\\28992\\Desktop\\work\\test_data\\lung2\\lung2_image.nrrd"
  # maskName = r"C:\\Users\\28992\\Desktop\\work\\test_data\\lung2\\lung2_label.nrrd"


  #5
  print(imageName)
  print(maskName)
  #6
  if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
      print('Error getting testcase!')
      exit()
  # Regulate verbosity with radiomics.verbosity
  # radiomics.setVerbosity(logging.INFO)

  # Get the PyRadiomics logger (default log-level = INFO
  logger = radiomics.logger
  logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

  # Write out all log entries to a file
  handler = logging.FileHandler(filename='C:\\Users\\28992\\Desktop\\work\\extract_feature\\testLog.txt', mode='w')
  formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  # Initialize feature extractor using the settings file
  extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
  featureClasses = getFeatureClasses()

  # Uncomment one of these functions to show how PyRadiomics can use the 'tqdm' or 'click' package to report progress when
  # running in full python mode. Assumes the respective package is installed (not included in the requirements)

  tqdmProgressbar()
  # clickProgressbar()

  print("Active features:")
  for cls, features in six.iteritems(extractor.enabledFeatures):
      if features is None or len(features) == 0:
          features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
      for f in features:
          print(f)
          print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)

  print("Calculating features")
  featureVector = extractor.execute(imageName, maskName, voxelBased=True)

  #7
for featureName, featureValue in six.iteritems(featureVector):
  if isinstance(featureValue, sitk.Image):
    sitk.WriteImage(featureValue, 'C:\\Users\\28992\\Desktop\\work\\extract_feature\\%s_%s.nrrd' % (testCase, featureName))
    print('Computed %s, stored as "%s_%s.nrrd"' % (featureName, testCase, featureName))
  else:
    print('%s: %s' % (featureName, featureValue))