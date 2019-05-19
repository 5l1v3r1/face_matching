# face_matching.py

After reading the great blog article series of @ageitgey I wanted to play around with some machine learning things.

Especially his blog post [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) gives a great introduction how face recognition works.

`face_matching.py` uses his python library "face_recognition" to play around with face recognition techniques.

You just need to pass a portrait of two persons to the script and a folder with some example pictures.
`face_matching.py` generates a log file with the results and opens each picture that has been passed to the script.
In this so-called preview mode, the script draws a box araound the recognized face and the name of person1 or person2.
Also the similarity is shown.

You can also disable the preview mode using the `--no_preview / -n` flag.


## Requirements / Dependencies

I recommend to create a python virtual environment for running `face_matching.py`.

```
Click==7.0
dlib==19.17.0
face-recognition==1.2.3
face-recognition-models==0.3.0
numpy==1.16.4
Pillow==6.0.0
```

## Arguments

```
usage: face_matching.py [-h] [-v] [-q] -p1 PERSON1 -p2 PERSON2 -i INPUT_FOLDER
                        [-l LOG] [-n]

uses face_recognition.py to play around with face recognition techniques.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Enable verbose logging.
  -q, --quiet           Disable all output. Only errors will be shown.
  -p1 PERSON1, --person1 PERSON1
                        Path to the first person that will be used for the
                        face matching.
  -p2 PERSON2, --person2 PERSON2
                        Path to the second person that will be used for the
                        face matching.
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Input folder with unknown images that should be
                        checked.
  -l LOG, --log LOG     Log file that should be used. If not set all output
                        will be logged to 'face_matching.log'
  -n, --no_preview      Deactivate picture generation.
```
