#!/usr/bin/env python3

import argparse
import face_recognition
import logging
import math
import numpy as np
import os
import sys
from PIL import Image, ImageDraw

logger = logging.getLogger("face_matching.py")

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    '''
    This functions converts the face distance to a percentage,
    Please note if you change the face match threshold from 0.6 (default) you must change this
    value in this function as well.
    '''
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def helper_get_images_from_folder(folder_name):
    '''
    This helper function walks recursive through each folder under the specified path
    and returns all image files that end with *.jpg in a list.
    '''
    images = []
    for root, dirs, files in os.walk(folder_name):
        for f in files:
            # check if file ends with *.jpg
            if f.endswith(".jpg"):
                # add the path from root to the image name to the images list
                images.append(os.path.join(root, f))
                # sort the images
                images.sort()

    # return the images list
    return images

def helper_check_file(file_path):
    '''
    This helper functions checks if an file_name exists and returns the file path,
    otherwise it will write an error log and returns None.
    '''
    exists = os.path.isfile(file_path)
    if exists:
        logger.debug("File: {} exists.".format(file_path))
        return file_path
    else:
        logger.error("File: {} does not exist. Please check file path.".format(file_path))
        return None

def helper_check_directory(path):
    '''
    This helper functions checks if an directory exists and returns its path, otherwise it will
    write an error log and returns None.
    '''
    exists = os.path.exists(path)
    if exists:
        logger.debug("Directory: {} exists.".format(path))
        return path
    else:
        logger.error("Directory: {} does not exist. Please check file path.".format(path))
        return None

def check_unknown_image(image_name, known_encodings, known_names,preview):
    '''
    Function checks an unknown image, builds the encoding and compares the new image
    to the known images. It draws an box around the identified faces with the similarity
    in percent.
    '''
    # load the image with the unknown face
    unknown_image = face_recognition.load_image_file(image_name)
    logger.info("----------------------------------------------------------------------------------------------------")
    logger.debug("Calculating similarity for unknown image: {}".format(image_name))

    # find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    if preview is True:
        # convert the image to a PIL-format so that we can draw on top of it using the Pillow library
        pil_image = Image.fromarray(unknown_image)
        # create a Pillow ImageDraw Draw instance to draw with it
        draw = ImageDraw.Draw(pil_image)

    # loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # check if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        # face_distance[0] is the distance between person 1 and unknown image
        person_1_distance = face_distances[0]
        # calculate similarity in percent for person 1
        person_1_similarity = face_distance_to_conf(person_1_distance)

        logger.debug("Face distance between person 1: {} and unknown image: {} is {} - Similarity: {:.4f} %".format(known_names[0], image_name, person_1_distance, person_1_similarity))
        # face_distance[1] is the distance between person 2 and unknown image
        person_2_distance = face_distances[1]
        # calculate similarity in percent for person 2
        person_2_similarity = face_distance_to_conf(person_2_distance)
        logger.debug("Face distance between person 2: {} and unknown image: {} is {} - Similarity: {:.4f} %".format(known_names[1], image_name, person_2_distance, person_2_similarity))

        # numpy.argmin returns the minimum value inside the face_distances ndarray pandas
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            logger.info("BEST-MATCH: Unknown image: {}".format(image_name))
            logger.info("BEST-MATCH: Eucledian distance: {}".format(face_distances[best_match_index]))
            match_percentage=face_distance_to_conf(face_distances[best_match_index])
            logger.info("BEST_MATCH: Similarity Confidence: :{:.2%}".format(match_percentage))
            logger.info("BEST_MATCH: Identified as: {}\n".format(name))

        if preview is True:
            # convert the image to a PIL-format so that we can draw on top of it using the Pillow library
            # draw a box arround the face unsing the Pillow Module
            draw.rectangle(((left, top), (right, bottom)), outline=(0,0,255))

            # draw a Label with the name below the face
            draw_string = "Identified as: "+name+" Similarity: "+f"{match_percentage:.4f} %"
            text_width, text_height = draw.textsize(draw_string)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), draw_string, fill=(255, 255, 255, 255))

            # remove the drawing library from memory as per the Pillow docs
            del draw

            # display the resulting image
            pil_image.show()


def main(argv=None):

    # prepare the arguments
    if argv is None:
        argv = sys.argv[1:]

    # initialisation of arguments
    parser = argparse.ArgumentParser(description='uses face_recognition.py to play around with face recognition techniques.')
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Disable all output. Only errors will be shown.")
    parser.add_argument("-p1", "--person1", type=str, help="Path to the first person that will be used for the face matching.",required=True)
    parser.add_argument("-p2", "--person2", type=str, help="Path to the second person that will be used for the face matching.", required=True)
    parser.add_argument("-i", "--input_folder", type=str, help="Input folder with unknown images that should be checked." ,required=True)
    parser.add_argument("-l", "--log", type=str, help="Log file that should be used. If not set all output will be logged to 'face_matching.log'",default="face_matching.log")
    parser.add_argument("-n", "--no_preview", action="store_false", help="Deactivate picture generation.")
    args = parser.parse_args(args=argv)

    if args.log:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(args.log)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)
        fh.setLevel(logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger().setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
    logger.debug('Starting face_matching.py...')
    logger.debug("Using log file: {}".format(args.log))


    # check if image file exists
    if helper_check_file(args.person1) is not None:
        # load picture of person_1 and learn how to recognize it
        person_1_image = face_recognition.load_image_file(args.person1)
        logger.debug("Loading picture of person 1 {}".format(args.person1))
        person_1_encoding = face_recognition.face_encodings(person_1_image)[0]
    else:
        return -1

    if helper_check_file(args.person2) is not None:
        # load picture of person_2 and learn how to recognize it
        person_2_image = face_recognition.load_image_file(args.person2)
        logger.debug("Loading picture of person 2 {}".format(args.person2))
        person_2_encoding = face_recognition.face_encodings(person_2_image)[0]
    else:
        return -1

    # create an array of known face encodings and their file names
    known_face_encodings = [
        person_1_encoding,
        person_2_encoding
    ]

    # use the filenames as kown_face_names
    #TODO: maybe add a settings file with names and filenames
    known_face_names = [
        os.path.basename(args.person1),
        os.path.basename(args.person2)
    ]
    logger.debug("Known face name of person 1: {}".format(os.path.basename(args.person1)))
    logger.debug("Known face name of person 2: {}".format(os.path.basename(args.person2)))

    # check if the input_folder exists
    input_folder = helper_check_directory(args.input_folder)
    if input_folder is not None:
        # use our helper function to get all the unknown images of the args.input_folder
        logger.debug("Loading the unknown images from path: {}".format(input_folder))
        unknown_images = helper_get_images_from_folder(folder_name=input_folder)

        # try to recognize the unknwon images (test just one)
        # check_unknown_image(unknown_images[0], known_face_encodings, known_face_names)
        # try to recognize the unknwon images (all)
        for u in unknown_images:
            check_unknown_image(u, known_face_encodings, known_face_names,args.no_preview)
        return 0

    else:
        return -1

if __name__ == "__main__":
    sys.exit(main())
