# Vehicle Counting and Identification
 
[![Compiled output of the project](https://img.youtube.com/vi/r_tIaplsZXA/0.jpg)](https://www.youtube.com/watch?v=r_tIaplsZXA)

The project is basically for the identification and recognition of vehicle number plates using deep learning. This project is trained on **standard Pakistan number plates**.

Following features are available at this time:


*   **Identification and recognition of number plate**
*   **Counting of Vehicles**

The models was trained and programmed in a way that it only recognizes vehicles from the front side. So, one can utilize this project to count number of vehicles in a closed area (by counting).

## Requirements:
The project uses following libraries:


*   Tesserect
*   Open CV (CV2)
*   PIL
*   ... (if I'm missing anything just install 😊)

If I'm finding time in future hopefully I'll create a `requirements.txt`

## Usage:
At this moment the project takes pictures from the `/pics` folder (assuming they are ordered) pass them through the trained models and show the recognized number plates.

One can modify the project to take pictures from a camera or video file directly.

Once the `pics` folder has your video frames just run `init_with_classes.py` 😊

## Any Issue?
The main purpose of this repo is to help someone working of similar project. So, just post an issue and I'll try to help you.