## Face Detection

### Requirements

- Python 3.3+ or Python 2.7
- macOS or Linux 

**Detect if picture has a face**

Requirements

    sudo apt-get install -y cmake libsm6 libxext6 libxrender-dev python-matplotlib python-tk

Install

    pip install -r requirements.txt
    
    or
    
    pip install Pillow matplotlib face_recognition keras tensorflow opencv-python

Usage:

    python detect.py path_to_image/picture.jpeg
    
Returns

    false or or emotion detection prediction (happy, sad, angry, etc)
