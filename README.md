# Train Yolo Model

This project explores training and using a YOLO (You Only Look Once) model for object detection, focusing on classifying facial expressions from a custom dataset. To create a more personalized dataset, I captured and labeled around 1500 images of my own facial expressions using Label Studio. It was more of a personal, self-tailored project, and it was a great way to learn and get familiar with YOLO in a practical setting. This file primarily discusses facial expressions, but the approach can be adapted to any other labeling task.

## Table of Contents
- [Project Overview](#project-overview)
- [Train With Google Colab](#train-with-google-colab)
- [Project Structure](#project-structure)
- [Deploying the Model on a PC](#deploying-the-model-on-a-pc)
- [Inspiration & Future Improvements](#inspiration--future-improvements)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview

### 1. Creating Custom Dataset & Manual Labeling

While many pre-labeled datasets are available online, I chose to take my own pictures to create a personalized dataset. I focused on classifying my own facial expressions—such as Natural, Happy, and Surprised—to add a personal touch to the project.  
- **Manual Labeling:** The manual labeling process, while time-consuming and requiring patience, was refreshing and allowed me to personalize the project. It took a few hours to label all the pictures.
- **Label Studio:** I used [Label Studio](https://labelstud.io/) for labeling, which provided a flexible interface for annotating my images.

### 2. YOLO Model Training

The training process involves:
- Preparing the custom dataset with labeled images.
- Configuring the YOLO model with appropriate parameters.
- Training the model using the YOLO framework to detect and classify the facial expressions.

### 3. Object Detection & Statistics Collection

After training, the model is used to:
- Detect facial expressions in webcam feeds and videos.
- Count the number of detections for each expression in each session.
- Save these session statistics to a CSV file (`face_expression_stats.csv`) for further analysis.

### 4. Visualizing the Results

The project includes a visualization component:
- A script (`view_stats.py`) reads the CSV file and generates a pie chart displaying the distribution of detected expressions across all sessions.

## **Train With Google Colab**
Click below to open the Google Colab notebook and start training your YOLO model with just a few steps. With Google Colab, you can upload an image dataset and start training with just a few lines of code.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QEWZK5emoj3fvD9SPrjAeCRPV5COxCy7?usp=sharing)


## Project Structure

- **`yolo_detect.py`**: The main script for running object detection using the trained YOLO model. It loads the model, processes input sources (images, videos, or live camera streams), performs detection, counts and saves the detection statistics.
- **`view_stats.py`**: A script for visualizing the collected statistics. It reads the CSV file with detection statistics and generates visualizations to analyze the results.
- **`face_expression_stats.csv`**: A CSV file that stores the statistics of detected facial expressions.
- **`.gitignore`**: Specifies which files and directories to ignore in the Git repository.
- **`README.md`**: Providing an overview of the project and its usage.

## Deploying the Model on a PC

The easiest way to run Ultralytics YOLO models on a PC is using **Anaconda**, which helps set up a virtual Python environment with all necessary dependencies.

### 1. Prerequisites

#### **1.1. Verify Python Version**

Ensure you have **Python 3.X** installed. You can check your current Python version with:
```bash
python -V
```

Python 3.8+ is recommended

If you need to install Python, download and install from [python.org](https://www.python.org/downloads/)

#### **1.2. Install Anaconda**

[Download Anaconda](https://anaconda.com/download) and follow the default installation settings.


#### **1.3. Set up a virtual environment**
Open **Anaconda Prompt** (or a terminal on macOS/Linux) and create a new Python environment:
```bash
conda create --name yolo_env python=3.X -y # Replace 3.X with your python version
conda activate yolo_env
```
Once inside the Conda environment, install the necessary Python packages:

```bash
pip install ultralytics opencv-python pandas matplotlib numpy
```

For NVIDIA GPU users, install the GPU-accelerated version of PyTorch. Be sure to check the latest compatible CUDA version on the [PyTorch website](https://pytorch.org/).

For example:
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Extract the trained model

#### Option 1: Download your model and unzip
Unzip ```my_model.zip``` and **navigate into its folder**:
```bash
cd path/to/folder
```
#### Option 2: Download my pretrained model
If you'd like, you can download my trained facial expressions model and run it.

* First, create a folder for the project

* Click on ```my_model.pt```, then select 'Download raw file'.

* Move the file into the created folder.

Once downloaded, you're ready for the next step.

### 3. Download and run yolo_detect.py
#### 3.1. Download the detection script:

```bash
curl -o yolo_detect.py https://raw.githubusercontent.com/Etaizil/Train_Yolo_Model/refs/heads/main/yolo_detect.py
```

#### 3.2. Download view_stats.py:

```bash
curl -o view_stats.py https://raw.githubusercontent.com/Etaizil/Train_Yolo_Model/refs/heads/main/view_stats.py
``` 

#### 3.3. Run inference with a YOLO model on a laptop or USB camera at 1280x720 resolution:

Using a webcam:
```bash
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720
```
Or using a laptop built-in camera if available:
```bash
python yolo_detect.py --model my_model.pt --source laptop0 --resolution 1280x720
```

* When prompted, enter a session name and press Enter.
* A window will display a live feed with bounding boxes drawn around detected facial expressions.

Alternatively, to process an image, video, or folder of images, specify the source:

```bash
python yolo_detect.py --model my_model.pt --source path/to/image_or_video.mp4
```

And that's it! Your camera is up and running to detect what it was trained for.

### 4. View Stats

Review detection statistics from all recorded sessions:
```bash
python view_stats.py
```

## Inspiration & Future Improvements

This project was inspired in part by the ideas and techniques presented in this [YouTube video](https://www.youtube.com/watch?v=r0RspiLG260), which provided valuable guidance as I developed my workflow.  
While the project is currently a proof-of-concept for myself, there are plenty of opportunities for improvement:
- **Scaling Up:** Manual labeling of large datasets is time-consuming. Integrating semi-automated labeling tools could make this process more efficient.
- **Model Enhancements:** I only did a single training run for this project. Further tuning of the YOLO model and experimenting with different architectures could improve detection accuracy.
- **Extended Datasets:** Combining my custom dataset with pre-existing datasets might boost the model's performance and generalizability.

## Conclusion

This project is a personal, hands-on exploration of training a YOLO model for object detection, with a unique focus on classifying my own facial expressions. While there's room for improvement, the manual approach provided a valuable learning experience and allowed me to tailor the dataset to my needs. Whether you're interested in custom model training or simply exploring computer vision techniques, this project offers a practical example of what's possible with dedication and creativity.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
