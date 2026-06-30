import os


def create_output_directory():

    os.makedirs(r"C:\Users\arman\OneDrive\Desktop\preprocess", exist_ok=True)


def confidence_percentage(confidence):

    return round(confidence * 100, 2)