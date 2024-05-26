from cannyEdgeDetecor import *
from harrisCornerDetector import *
from houghLineTransfrom import *
from hough2 import *
from eigenfaces import *
from eigenfacesNew import *
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage


class ImageLabel(QLabel):
    index = 0

    def __init__(self, indx, parent=None):
        self.index = indx
        super().__init__(parent)

    def mouseDoubleClickEvent(self, event):
        displayImages(imagePaths, self.index)


def displayImages(imagePaths, index):
    path = QFileDialog.getOpenFileName(
        None, "Open Image File", "", "Image Files (*.jpg)")
    if path:
        if tabWidget.currentIndex() == 0:
            imagePaths[index] = path[0]
            pixmap = QPixmap(imagePaths[index]).scaled(
                400, 400, aspectRatioMode=Qt.KeepAspectRatio)
            imageContainers[index].setPixmap(pixmap)
            altered = cannyEdgeDetector(cv2.imread(
                imagePaths[index]), inputK, inputSigma, inputLowT, inputHighT)
            altered = cv2.resize(altered, (400, 400))
            cv2.imwrite('blank_image.png', altered)
            pixmap2 = QPixmap('blank_image.png')
            imageContainers[index+6].setPixmap(pixmap2)
        elif tabWidget.currentIndex() == 1:
            imagePaths[index] = path[0]
            pixmap = QPixmap(imagePaths[index]).scaled(
                600, 600, aspectRatioMode=Qt.KeepAspectRatio)
            imageContainers[index].setPixmap(pixmap)
            altered = houghLineTransform(cv2.imread(
                imagePaths[index]), inputNLines, inputRho, inputTheta)
            altered = cv2.resize(altered, (600, 600))
            cv2.imwrite('blank_image.png', altered)
            pixmap2 = QPixmap('blank_image.png')
            imageContainers[index+6].setPixmap(pixmap2)
        elif tabWidget.currentIndex() == 2:
            imagePaths[index] = path[0]
            pixmap = QPixmap(imagePaths[index]).scaled(
                600, 600, aspectRatioMode=Qt.KeepAspectRatio)
            imageContainers[index].setPixmap(pixmap)
            altered = harrisCornerDetector(cv2.imread(
                imagePaths[index]), inputHarThr)
            altered = cv2.resize(altered, (600, 600))
            cv2.imwrite('blank_image.png', altered)
            pixmap2 = QPixmap('blank_image.png')
            imageContainers[index+6].setPixmap(pixmap2)
        elif tabWidget.currentIndex() == 3:
            imagePaths[index] = path[0]
            pixmap = QPixmap(imagePaths[index]).scaled(
                600, 600, aspectRatioMode=Qt.KeepAspectRatio)
            imageContainers[index].setPixmap(pixmap)
            altered = houghEllipse(cv2.imread(
                imagePaths[index]), inputEllK, inputEllSig, inputEllAlow, inputEllAHigh, inputEllBlow, inputEllBhigh)
            # altered = cv2.resize(altered, (600, 600))
            cv2.imwrite('blank_image.png', altered)
            pixmap2 = QPixmap('blank_image.png').scaled(
                600, 600, aspectRatioMode=Qt.KeepAspectRatio)
            imageContainers[index+6].setPixmap(pixmap2)


# Canny Setters
def setSigma(sig):
    global inputSigma
    inputSigma = sig


def setK(k):
    global inputK
    inputK = k


def setLowT(lowT):
    global inputLowT
    inputLowT = lowT


def setHighT(highT):
    global inputHighT
    inputHighT = highT


def setAllCanny():
    setSigma(int(sigma.text()))
    setK(int(kernelSize.text()))
    setLowT(int(lowThreshold.text()))
    setHighT(int(highThreshold.text()))

# Hough Setters


def setNLinesHough(nLines):
    global inputNLines
    inputNLines = nLines


def setRhoHough(rho):
    global inputRho
    inputRho = rho


def setThetaHough(theta):
    global inputTheta
    inputTheta = theta


def setAllHough():
    setNLinesHough(int(nLinesHough.text()))
    setRhoHough(float(rhoHough.text()))
    setThetaHough(float(thetaHough.text()))

# Harris Setters


def setHarrisThres(harrT):
    global inputHarThr
    inputHarThr = harrT

# Ellipse Setters


def setEllK(ellK):
    global inputEllK
    inputEllK = ellK


def setEllSig(ellSig):
    global inputEllSig
    inputEllSig = ellSig


def setEllAlow(ellAlow):
    global inputEllAlow
    inputEllAlow = ellAlow


def setEllAhigh(ellAhigh):
    global inputEllAHigh
    inputEllAHigh = ellAhigh


def setEllBlow(ellBlow):
    global inputEllBlow
    inputEllBlow = ellBlow


def setEllBhigh(ellBhigh):
    global inputEllBhigh
    inputEllBhigh = ellBhigh


def setAllEll():
    setEllK(int(EllipseK.text()))
    setEllSig(int(EllipseSigma.text()))
    setEllAlow(int(EllipseALow.text()))
    setEllAhigh(int(EllipseAHigh.text()))
    setEllBlow(int(EllipseBLow.text()))
    setEllBhigh(int(EllipseBHigh.text()))

# Eigen Setters


def setEigenTopK(topK):
    global inputEigenNumTopK
    inputEigenNumTopK = topK


def setEigenImageNum(imageNum):
    global inputEigenImageNum
    inputEigenImageNum = imageNum


def setAllEigen():
    setEigenTopK(int(numTopK.text()))
    setEigenImageNum(int(imageNumber.text()))
    print(inputEigenNumTopK, inputEigenImageNum)
    eigenFaces(inputEigenNumTopK, inputEigenImageNum)
    
def setAllEigenNew():
    setEigenTopK(int(numTopK.text()))
    setEigenImageNum(int(imageNumber.text()))
    print(inputEigenNumTopK, inputEigenImageNum)
    eigenFacesNew(inputEigenNumTopK, inputEigenImageNum)


# Create a PyQt5 application
app = QApplication(sys.argv)

inputK = 9
inputSigma = 3
inputLowT = 10
inputHighT = 30

inputHarThr = 0.01

inputNLines = 8
inputRho = 1
inputTheta = 1

inputEllK = 5
inputEllSig = 3
inputEllAlow = 1
inputEllAHigh = 2
inputEllBlow = 1
inputEllBhigh = 2

inputEigenNumTopK = 40
inputEigenImageNum = 0

# Create a window
tabWidget = QTabWidget()
tab1 = QWidget()
labelK = QLabel("K: ")
labelSigma = QLabel("Sigma: ")
labelHighT = QLabel("High Threshold: ")
labelLowT = QLabel("Low Threshold: ")

spacerItem = QWidget()
spacerItem.setMinimumSize(120, 0)
spacerItem.setMaximumSize(120, 20)

sigma = QLineEdit("", None)
sigma.setFixedSize(120, 20)


kernelSize = QLineEdit("", None)
kernelSize.setFixedSize(120, 20)


lowThreshold = QLineEdit("", None)
lowThreshold.setFixedSize(120, 20)

highThreshold = QLineEdit("", None)
highThreshold.setFixedSize(120, 20)
submitAllCanny = QPushButton("Submit")
submitAllCanny.clicked.connect(
    lambda: setAllCanny())

tabWidget.setWindowTitle('Task 3')
tabWidget.setGeometry(100, 100, 1250, 825)  # (x, y, width, height)

parentLayout = QVBoxLayout()
originalImageLayout = QHBoxLayout()
alteredImageLayout = QHBoxLayout()

toolBarCanny = QToolBar()
toolBarCanny.setContentsMargins(10, 0, 10, 0)

left_spacer = QWidget()
left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

right_spacer = QWidget()
right_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

toolBarCanny.addWidget(left_spacer)

toolBarCanny.addWidget(labelK)
toolBarCanny.addWidget(kernelSize)
toolBarCanny.addSeparator()

toolBarCanny.addWidget(labelSigma)
toolBarCanny.addWidget(sigma)
toolBarCanny.addSeparator()

toolBarCanny.addWidget(labelLowT)
toolBarCanny.addWidget(lowThreshold)
toolBarCanny.addSeparator()

toolBarCanny.addWidget(labelHighT)
toolBarCanny.addWidget(highThreshold)
toolBarCanny.addWidget(spacerItem)

toolBarCanny.addWidget(submitAllCanny)
toolBarCanny.addWidget(right_spacer)

imagePaths = ["", "", "", "", "", ""]

tabWidget.addTab(tab1, "Canny Edge Detector")


tab1.setLayout(parentLayout)
parentLayout.addWidget(toolBarCanny)
parentLayout.addLayout(originalImageLayout)
parentLayout.addLayout(alteredImageLayout)

originalImage1 = ImageLabel(0)
originalImage2 = ImageLabel(1)
originalImage3 = ImageLabel(2)
originalImageLayout.addWidget(originalImage1)
originalImageLayout.addWidget(originalImage2)
originalImageLayout.addWidget(originalImage3)

alteredImage1 = ImageLabel(0)
alteredImage2 = ImageLabel(1)
alteredImage3 = ImageLabel(2)
alteredImageLayout.addWidget(alteredImage1)
alteredImageLayout.addWidget(alteredImage2)
alteredImageLayout.addWidget(alteredImage3)


# Tab 2
tab2 = QWidget()

tabWidget.addTab(tab2, "Hough Line")
originalImage4 = ImageLabel(3)
alteredImage4 = ImageLabel(3)

parentLayoutHough = QVBoxLayout()
imageLayoutHough = QHBoxLayout()
toolBarHough = QToolBar()
toolBarHough.setContentsMargins(10, 0, 10, 0)
spacerItemHough = QWidget()
spacerItemHough.setMinimumSize(120, 0)
spacerItemHough.setMaximumSize(120, 20)
left_spacerHo = QWidget()
left_spacerHo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
right_spacerHo = QWidget()
right_spacerHo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

nLinesHough = QLineEdit("", None)
nLinesHough.setFixedSize(120, 20)
labelNLines = QLabel("Number of Lines: ")

rhoHough = QLineEdit("", None)
rhoHough.setFixedSize(120, 20)
setAllHoughButton = QPushButton("Submit")
setAllHoughButton.clicked.connect(
    lambda: setAllHough())
labelRho = QLabel("Rho Resolution: ")

thetaHough = QLineEdit("", None)
labelTheta = QLabel("Theta Resolution: ")


toolBarHough.addWidget(left_spacerHo)
toolBarHough.addWidget(labelNLines)
toolBarHough.addWidget(nLinesHough)
toolBarHough.addSeparator()

toolBarHough.addWidget(labelRho)
toolBarHough.addWidget(rhoHough)
toolBarHough.addSeparator()
toolBarHough.addWidget(labelTheta)
toolBarHough.addWidget(thetaHough)

toolBarHough.addWidget(spacerItemHough)
toolBarHough.addWidget(setAllHoughButton)
toolBarHough.addWidget(right_spacerHo)

parentLayoutHough.addWidget(toolBarHough)
imageLayoutHough.addWidget(originalImage4)
imageLayoutHough.addWidget(alteredImage4)
parentLayoutHough.addLayout(imageLayoutHough)

tab2.setLayout(parentLayoutHough)


# Tab 3
tab3 = QWidget()
tabWidget.addTab(tab3, "Harris Corner")

originalImage5 = ImageLabel(4)
alteredImage5 = ImageLabel(4)
parentLayoutHarris = QVBoxLayout()
imageLayoutHarris = QHBoxLayout()
toolBarHarris = QToolBar()
toolBarHarris.setContentsMargins(10, 0, 10, 0)

left_spacerH = QWidget()
left_spacerH.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
right_spacerH = QWidget()
right_spacerH.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


thresholdHarris = QLineEdit("", None)
thresholdHarris.setFixedSize(120, 20)
thresholdHarrisButton = QPushButton("Submit")
thresholdHarrisButton.clicked.connect(
    lambda: setHarrisThres(float(thresholdHarris.text())))
labelThresholdH = QLabel("Threshold: ")

toolBarHarris.addWidget(left_spacerH)
toolBarHarris.addWidget(labelThresholdH)
toolBarHarris.addWidget(thresholdHarris)
toolBarHarris.addWidget(thresholdHarrisButton)
toolBarHarris.addWidget(right_spacerH)


parentLayoutHarris.addWidget(toolBarHarris)
imageLayoutHarris.addWidget(originalImage5)
imageLayoutHarris.addWidget(alteredImage5)
parentLayoutHarris.addLayout(imageLayoutHarris)

tab3.setLayout(parentLayoutHarris)


# tab 4
tab4 = QWidget()
tabWidget.addTab(tab4, "Hough Ellipse")

originalImage6 = ImageLabel(5)
alteredImage6 = ImageLabel(5)

parentLayoutEllipse = QVBoxLayout()
imageLayoutEllipse = QHBoxLayout()
toolBarEllipse = QToolBar()
toolBarEllipse.setContentsMargins(10, 0, 10, 0)

left_spacerEllipse = QWidget()
left_spacerEllipse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
right_spacerEllipse = QWidget()
right_spacerEllipse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

EllipseK = QLineEdit("", None)
EllipseK.setFixedSize(120, 20)
LabelEllipseK = QLabel("k: ")

EllipseSigma = QLineEdit("", None)
EllipseSigma.setFixedSize(120, 20)
labelEllipseSigma = QLabel("Sigma: ")

EllipseALow = QLineEdit("", None)
EllipseALow.setFixedSize(120, 20)
labelEllipseALow = QLabel("A low: ")

EllipseAHigh = QLineEdit("", None)
EllipseAHigh.setFixedSize(120, 20)
labelEllipseAHigh = QLabel("A high: ")

EllipseBLow = QLineEdit("", None)
EllipseBLow.setFixedSize(120, 20)
labelEllipseBLow = QLabel("B low: ")

EllipseBHigh = QLineEdit("", None)
EllipseBHigh.setFixedSize(120, 20)
labelEllipseBHigh = QLabel("B high: ")

setAllEllipse = QPushButton("Submit")
setAllEllipse.clicked.connect(
    lambda: setAllEll())

toolBarEllipse.addWidget(left_spacerEllipse)

toolBarEllipse.addWidget(LabelEllipseK)
toolBarEllipse.addWidget(EllipseK)
toolBarEllipse.addSeparator()

toolBarEllipse.addWidget(labelEllipseSigma)
toolBarEllipse.addWidget(EllipseSigma)
toolBarEllipse.addSeparator()

toolBarEllipse.addWidget(labelEllipseALow)
toolBarEllipse.addWidget(EllipseALow)
toolBarEllipse.addSeparator()

toolBarEllipse.addWidget(labelEllipseAHigh)
toolBarEllipse.addWidget(EllipseAHigh)
toolBarEllipse.addSeparator()

toolBarEllipse.addWidget(labelEllipseBLow)
toolBarEllipse.addWidget(EllipseBLow)
toolBarEllipse.addSeparator()

toolBarEllipse.addWidget(labelEllipseBHigh)
toolBarEllipse.addWidget(EllipseBHigh)
toolBarEllipse.addSeparator()

toolBarEllipse.addWidget(setAllEllipse)

toolBarEllipse.addWidget(right_spacerEllipse)

parentLayoutEllipse.addWidget(toolBarEllipse)
imageLayoutEllipse.addWidget(originalImage6)
imageLayoutEllipse.addWidget(alteredImage6)
parentLayoutEllipse.addLayout(imageLayoutEllipse)
tab4.setLayout(parentLayoutEllipse)

imageContainers = [originalImage1, originalImage2,
                   originalImage3, originalImage4, originalImage5, originalImage6, alteredImage1, alteredImage2, alteredImage3, alteredImage4, alteredImage5, alteredImage6]

# tab5
tab5 = QWidget()
tabWidget.addTab(tab5, "Eigen Faces")

parentLayoutEigen = QVBoxLayout()
imageLayoutEigen = QHBoxLayout()

toolBarEigen = QToolBar()
toolBarEigen.setContentsMargins(10, 0, 10, 0)

left_spacerEigen = QWidget()
left_spacerEigen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
right_spacerEigen = QWidget()
right_spacerEigen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

numTopK = QLineEdit("", None)
numTopK.setFixedSize(120, 20)
LabelNumTopK = QLabel("Top K Eigen Faces: ")

imageNumber = QLineEdit("", None)
imageNumber.setFixedSize(120, 20)
imageNumberLabel = QLabel("Image number 0-40: ")

setAllEigenButton = QPushButton("Submit")
setAllEigenButton.setFixedSize(120, 20)
setAllEigenButton.clicked.connect(
    lambda: setAllEigen())

setAllEigenNewButton = QPushButton("Submit New")
setAllEigenNewButton.setFixedSize(120, 20)
setAllEigenNewButton.clicked.connect(
    lambda: setAllEigenNew())

toolBarEigen.addWidget(left_spacerEigen)

toolBarEigen.addWidget(LabelNumTopK)
toolBarEigen.addWidget(numTopK)
toolBarEigen.addSeparator()

toolBarEigen.addWidget(imageNumberLabel)
toolBarEigen.addWidget(imageNumber)
toolBarEigen.addWidget(setAllEigenButton)
toolBarEigen.addWidget(setAllEigenNewButton)


toolBarEigen.addWidget(right_spacerEigen)


parentLayoutEigen.addWidget(toolBarEigen)
parentLayoutEigen.addLayout(imageLayoutEigen)

tab5.setLayout(parentLayoutEigen)


# Show the window
tabWidget.show()
# Start the event loop
sys.exit(app.exec_())
