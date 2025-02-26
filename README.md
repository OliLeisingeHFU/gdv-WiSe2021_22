# gdv-WiSe2021_22
Exercises for course "Grafische Datenverarbeitung" in winter term 2021 at HS Furtwangen University. Please note that the lecture and the accompanying tutorials (Übung) is not defined to the end. All content here is tentative and will be adopted to the students' needs during the course.

# Prerequisites
- Install Python 3 from python.org
  - Version 3.9.7 is recommended, Python 3.10 does not work (yet)
  - Test which version is installed on your machine with `python --version` in the terminal
    - Ensure that the correct version is used in the terminal as well as the selected interpreter: ![Screenshot with correct python interpreter selected](install/pyenv_hint.png)
- Install pip from https://pip.pypa.io/en/stable/installation/
  - Ensure that a version newer than 19.3 is installed


## Further Python ressources
If you are not familiar with Python, check out the following tutorials:
- https://www.python.org/about/gettingstarted/ 
- https://docs.python.org/3/tutorial/
- https://code.visualstudio.com/docs/python/python-tutorial


## VS Code
The code is all developed using VS Code. You can use any IDE, but VS Code will be used in the lecture.
- Install VS Code from https://code.visualstudio.com/

---
# Installing OpenCV
- Install opencv as pip module opencv-python as explained on https://pypi.org/project/opencv-python/ (Main modules should be enough for the beginning)

| Windows         | MacOS     | Linux |
|--------------|-----------|------------|
|py -m pip install opencv-python|python -m pip install opencv-python|python -m pip install opencv-python|

⚠ Note that there is no need to install OpenCV from opencv.org

⚠ Ensure that Python 3.9.7 is used as mentioned above.

See https://docs.opencv.org/master/d0/de3/tutorial_py_intro.html for further help on other systems.

---
# Helpful ressources

## Python
You can use https://docs.python.org/3/ as a starting point and the [Library Reference](https://docs.python.org/3/library/index.html) or the [Language Reference](https://docs.python.org/3/reference/index.html) should contain all the needed information.

## OpenCV reference
See https://docs.opencv.org/4.5.3/ for the OpenCV code reference. Here, all OpenCV methods are explained. If you want to know about parameters and flags, this is the page to look them up. 

## NumPy
OpenCV uses NumPy ndarrays as the common format for data exchange. It can create, operate on, and work with NumPy arrays. For some operations it makes sense to import the NumPy module and use special functions provided by NumPy. Other libraries like TensorFlow and SciPy also use NumPy. See https://numpy.org/doc/stable/reference/index.html for the API reference.

## Python style guide
All these tutorials are written according to the [PEP8 Python Code Style Guide](https://www.python.org/dev/peps/pep-0008/). This is realized using the Python tools [pycodestyle (pep8)](https://code.visualstudio.com/docs/python/linting#_pycodestyle-pep8) and [autopep8](https://pypi.org/project/autopep8/).

## Other tutorials
- [Tech with Tim: OpenCV Python Tutorials](https://www.youtube.com/watch?v=qCR2Weh64h4&list=PLzMcBGfZo4-lUA8uGjeXhBUUzPYc6vZRn)
- [freeCodeCamp.org -OpenCV Course - Full Tutorial with Python](https://www.youtube.com/watch?v=oXlwWbU8l2o) (not yet watched)

---
# Tutorials

## Tutorial #1
Load, resize and rotate an image. And display it to the screen.
- [empty code](./GDV_tutorial_01_empty.py)
- [complete code](./GDV_tutorial_01.py)

## Tutorial #2
Direct pixel access and manipulation. Set some pixels to black, copy some part of the image to some other place, count the used colors in the image
- [empty code](./GDV_tutorial_02_empty.py)
- [complete code](./GDV_tutorial_02.py)

## Tutorial #3
Show camera video and mirror it.
- [empty code](./GDV_tutorial_03_empty.py)
- [complete code](./GDV_tutorial_03.py)

## Tutorial #4
Loading a video file and mirror it.
- [empty code](./GDV_tutorial_04_empty.py)
- [complete code](./GDV_tutorial_04.py)

## Tutorial #5
Use the webcam image stream and draw something on it. Animate one of the drawings.
- [empty code](./GDV_tutorial_05_empty.py)
- [complete code](./GDV_tutorial_05.py)

## Tutorial #6
Playing around with colors. We convert some values from RGB to HSV and then find colored objects in the image and mask them out.
- [empty code](./GDV_tutorial_06_empty.py)
- [complete code](./GDV_tutorial_06.py)

## Tutorial #7
Counting colored objects by finding connected components in the binary image. Modify the binary image to improve the results.
- [empty code](./GDV_tutorial_07_empty.py)
- [complete code](./GDV_tutorial_07.py)

## Tutorial #8
Demonstrating how to do template matching in OpenCV. 
- [empty code](./GDV_tutorial_08_empty.py)
- [complete code](./GDV_tutorial_08.py)

## Tutorial #9
Demonstrating Gaussian blur filter with OpenCV. 
- [empty code](./GDV_tutorial_09_empty.py)
- [complete code](./GDV_tutorial_09.py)
- [complete code with 3D plot of the kernel using matplotlib](./GDV_tutorial_09_3Dplot.py)
  - Note that matplotlib needs to be installed as described [here](https://matplotlib.org/stable/users/installing.html)

## Tutorial #10
Doing the Fourier Transform for images and back.
- [empty code](./GDV_tutorial_10_empty.py)
- [complete code](./GDV_tutorial_10.py)

## Tutorial #11
Geometric transformations a.k.a. image warping.
- [empty code](./GDV_tutorial_11_empty.py)
- [complete code](./GDV_tutorial_11.py)

## Tutorial #12
Select three points in two images and compute the appropriate affine transformation.
- [empty code](./GDV_tutorial_12_empty.py)
- [complete code](./GDV_tutorial_12.py)

## Tutorial #13
Select four points in two images and compute the appropriate projective/perspective transformation.
- [empty code](./GDV_tutorial_13_empty.py)
- [complete code](./GDV_tutorial_13.py)

## Tutorial #14
Compute the edges of an image with the Canny edge detection. Adjust the parameters using sliders.
- [empty code](./GDV_tutorial_14_empty.py)
- [complete code](./GDV_tutorial_14.py)

## Tutorial #15
Compute the features of an image with the Harris corner detection. Adjust the parameters using sliders.
- [empty code](./GDV_tutorial_15_empty.py)
- [complete code](./GDV_tutorial_15.py)


