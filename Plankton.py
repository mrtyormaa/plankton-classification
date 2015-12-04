__author__ = 'mrtyormaa'
# Although there is another file roate.py, it is not being used.
# That was a part of the experiment to artificially boost the number of training data images.
# The strategy was rejected and this is the final single file.


# Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from skimage import measure
from skimage import morphology
import numpy
import mahotas
from sklearn.preprocessing import Imputer

import warnings

warnings.filterwarnings("ignore")

# Preprocess the images and get the regions and labels
def getProps(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = numpy.where(image > numpy.mean(image), 0., 1.0)

    # Dilate the image
    imdilated = morphology.dilation(imagethr, numpy.ones((4, 4)))

    # Create the label list
    label_list = measure.label(imdilated)
    region_list = measure.regionprops(label_list)
    return region_list


# Get hu moments
def getHuMoments(image):
    img = image.copy()
    return measure.moments_hu(measure.moments(img))

# All the geomatrical properties of the system
def getAllMeasureProperties(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = numpy.where(image > numpy.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, numpy.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)

    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    minor_axis_length = 0.0
    major_axis_length = 0.0
    area = 0.0
    convex_area = 0.0
    eccentricity = 0.0
    equivalent_diameter = 0.0
    euler_number = 0.0
    extent = 0.0
    filled_area = 0.0
    orientation = 0.0
    perimeter = 0.0
    solidity = 0.0
    centroid = [0.0,0.0]
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
        minor_axis_length = 0.0 if maxregion is None else maxregion.minor_axis_length
        major_axis_length = 0.0 if maxregion is None else maxregion.major_axis_length
        area = 0.0 if maxregion is None else maxregion.area
        convex_area = 0.0 if maxregion is None else maxregion.convex_area
        eccentricity = 0.0 if maxregion is None else maxregion.eccentricity
        equivalent_diameter = 0.0 if maxregion is None else maxregion.equivalent_diameter
        euler_number = 0.0 if maxregion is None else maxregion.euler_number
        extent = 0.0 if maxregion is None else maxregion.extent
        filled_area = 0.0 if maxregion is None else maxregion.filled_area
        orientation = 0.0 if maxregion is None else maxregion.orientation
        perimeter = 0.0 if maxregion is None else maxregion.perimeter
        solidity = 0.0 if maxregion is None else maxregion.solidity
        centroid = [0.0,0.0] if maxregion is None else maxregion.centroid

    return ratio,minor_axis_length,major_axis_length,area,convex_area,eccentricity,\
           equivalent_diameter,euler_number,extent,filled_area,orientation,perimeter,solidity, centroid[0], centroid[1]

# Get eccentricity
def getEccentricity(props):
    return numpy.mean([prop.eccentricity for prop in props])


# Get eccentricity
def getPerimeter(props):
    return numpy.mean([prop.perimeter for prop in props])

# Get Zernike Momentss
def getZernikeMoments(image):
    img = image.copy()
    return mahotas.features.zernike_moments(img, radius=20, degree=8)

# Get Linear Binary Patterns
def getLBP(image):
    img = image.copy()
    return  mahotas.features.lbp(img, radius=20, points=7, ignore_zeros=False)

def uint_to_float(img):
    return 1 - (img / numpy.float32(255.0))

# Get Parameter Free Threshold Adjacency Statistics
def getPFTA(image):
    img = image.copy()
    return mahotas.features.pftas(img)

# Get Haralick Features
def getHaralickTextures(image):
    img = image.copy()
    return mahotas.features.haralick(img, ignore_zeros=False, preserve_haralick_bug=False,
                                     compute_14th_feature=False).flatten()

# Remove NaNs from a ndarray
def removeNaNs(X):
    imp=Imputer(missing_values='NaN',strategy='median',axis=0)
    return imp.fit_transform(X)


# find the largest nonzero region
def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label]) * 1.0 / regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop


def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = numpy.where(image > numpy.mean(image), 0., 1.0)

    # Dilate the image
    imdilated = morphology.dilation(imagethr, numpy.ones((4, 4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr * label_list
    label_list = label_list.astype(int)

    region_list = measure.regionprops(label_list)
    maximumregion = getLargestRegion(region_list, label_list, imagethr)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maximumregion is None) and (maximumregion.major_axis_length != 0.0)):
        ratio = 0.0 if maximumregion is None else maximumregion.minor_axis_length * 1.0 / maximumregion.major_axis_length
    return ratio


# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("train", "*"))
                           ).difference(set(glob.glob(os.path.join("train", "*.*")))))

# Rescale the images and create the combined metrics and training labels
# get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
                continue
            numberofImages += 1

# Rescale the images and create the combined metrics and training labels

# Rescaling the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_rows = numberofImages  # one row for each image in the training dataset
num_features = imageSize + 1 + 7 + 25 + 20 + 54 + 52 + 15 # for all of the features

# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X = numpy.zeros((num_rows, num_features), dtype=float)

# y is the numeric class label
y = numpy.zeros(num_rows)

files = []

# Generate training data
i = 0
label = 0

# List of string of class names
namesClasses = list()

#  Test fo ----
arr = numpy.zeros(num_rows)

print("Reading images")

# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)

    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
                continue

            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)
            axisratio = getMinorMajorRatio(image)
            image = resize(image, (maxPixel, maxPixel))
            image_unit8 = image.astype('uint8')
            imageProps = getProps(image)
            huMoments = getHuMoments(image)

            # Store the rescaled image pixels and the axis ratio
            X[i, 0:imageSize] = numpy.reshape(image, (1, imageSize))
            X[i, imageSize] = axisratio

            # All Extra features to improve performance
            X[i, imageSize + 1:imageSize + 8] = getHuMoments(image)

            X[i, imageSize + 8:imageSize + 8 + 25] = getZernikeMoments(image)

            X[i, imageSize + 33:imageSize + 33 + 20] = getLBP(image)

            X[i, imageSize + 53:imageSize + 53 + 54] = getPFTA(image_unit8)

            X[i, imageSize + 107:imageSize + 107 + 52] = getHaralickTextures(image_unit8)

            X[i, imageSize + 159:imageSize + 159 + 15] = numpy.array(getAllMeasureProperties(image))

            # arr[i] = numpy.array(getAllMeasureProperties(image)).size
            # Store the classlabel
            y[i] = label
            i += 1

            # report progress for each 5% done
            report = [int((j + 1) * num_rows / 20.) for j in range(20)]
            if i in report:
                print(numpy.ceil(i * 100.0 / num_rows), "% done")
    label += 1

# print(numpy.unique(arr))
# exit()
# Remove NaNs from our data
print("Filtering Data to remove NaNs")
new_X = removeNaNs(X[:,imageSize:num_features])
X[:,imageSize:num_features] = new_X
print("Filtering Complete")

print("Training Data Started")
# n_estimators is the number of decision trees
# max_features also known as m_try is set to the default value of the square root of the number of features
clf = RF(n_estimators=100, n_jobs=4)
scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1)

print("Accuracy of all classes")
print(numpy.mean(scores))

# Get the probability predictions for computing the log-loss function
kf = KFold(y, n_folds=5)

# prediction probabilities number of samples, by number of classes
y_pred = y * 0
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    clf = RF(n_estimators=100, n_jobs=3)
    clf.fit(X_train, y_train)
    y_pred[test] = clf.predict(X_test)

print(classification_report(y, y_pred, target_names=namesClasses))
