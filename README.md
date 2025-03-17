# Train Yolo Model

This project explores training and using a YOLO (You Only Look Once) model for object detection, focusing on classifying facial expressions from a custom dataset. To create a more personalized dataset, I captured and labeled around 1500 images of my own facial expressions using Label Studio. It was more of a personal, self-tailored project, and it was a great way to learn and get familiar with YOLO in a practical setting. This file will discuss mostly about 'facial expressions', but it can be derived to any other labeling idea.

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
- **Manual Labeling:** The manual process, though a bit time-consuming and requiring patience, felt refreshing and allowed me to make the project truly my own. It took a few hours to label all the pictures.
- **Label Studio:** I used Label Studio for labeling, which provided a flexible interface for annotating my images.

### 2. YOLO Model Training

The training process involves:
- Preparing the custom dataset with labeled images.
- Configuring the YOLO model with appropriate parameters.
- Training the model using the YOLO framework to detect and classify the facial expressions.

### 3. Object Detection & Statistics Collection

After training, the model is used to:
- Detect face expressions in webcam, images and videos.
- Count the number of detections for each expression in each session.
- Save these session statistics to a CSV file (`face_expression_stats.csv`) for further analysis.

### 4. Visualizing the Results

The project includes a visualization component:
- A script (`view_stats.py`) reads the CSV file and generates a pie chart that shows the distribution of expressions over all sessions.

## **Train With Google Colab**
Click below to open the Colab notebook and start training a YOLO model. This makes it easy to upload an image dataset and run a few blocks of code to get started.

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
If you need to install Python, download and install from [python.org](https://www.python.org/downloads/)

#### **1.2. Install Anaconda**

[Download Anaconda](https://anaconda.com/download) and follow the default installation settings.


#### **1.3. Set up a virtual environment**
Open **Anaconda Prompt** (or a terminal on macOS/Linux) and create a new Python environment:
```bash
conda create --name yolo_env python=3.X -y # Replace 3.X with your python version
conda activate yolo_env
```

Then, install Ultralytics:

```bash
pip install ultralytics
```

For NVIDIA GPU users, install the GPU-accelerated version of PyTorch:

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
If you want to, you may download my face expressions trained model and run it.

* First, create a folder for the project

* Simply click ```my_model.pt``` and select 'Download raw file'.

* Move the file into the created folder.

You may now proceed to the next step.

### 3. Download and run yolo_detect.py
#### 3.1. Download the detection script:

```bash
curl -o yolo_detect.py https://raw.githubusercontent.com/Etaizil/Train_Yolo_Model/refs/heads/main/yolo_detect.py
```

#### 3.2. Download view_stats.py:

```bash
curl -o view_stats.py https://raw.githubusercontent.com/Etaizil/Train_Yolo_Model/refs/heads/main/view_stats.py
``` 

#### 3.3. Run inference with a YOLOv8n model on a laptop or USB camera at 1280x720 resolution:

```bash
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720
```
or:
```bash
python yolo_detect.py --model my_model.pt --source laptop0 --resolution 1280x720
```

* Enter a session name ans press Enter.
* A window will display a live feed with bounding boxes drawn around detected facial expressions.

Alternatively, to process an image, video, or folder of images, specify the source:

```bash
python yolo_detect.py --model my_model.pt --source path/to/image_or_video.mp4
```

And that's it! Your camera is up and running to detect what it was trained for.

### 4. View Stats

Watch statistics from all sessions combined:
```bash
python view_stats.py
```

## Inspiration & Future Improvements

This project was partly inspired by the ideas and techniques presented in this [YouTube video](https://www.youtube.com/watch?v=r0RspiLG260), which provided valuable guidance as I developed my workflow.  
While the project is currently a proof-of-concept for myself, there are plenty of opportunities for improvement:
- **Scaling Up:** Manual labeling of large datasets is time-consuming. Integrating semi-automated labeling tools could make this process more efficient.
- **Model Enhancements:** I only did a single training run for this project. Further tuning of the YOLO model and experimenting with different architectures could improve detection accuracy.
- **Extended Datasets:** Combining my custom dataset with pre-existing datasets might boost the model's performance and generalizability.

## Conclusion

This project is a personal, hands-on exploration of training a YOLO model for object detection, with a unique focus on classifying my own facial expressions. Although there is ample room for improvement - the manual approach provided a valuable learning experience and allowed me to tailor the dataset to my specific needs. Whether you're interested in custom model training or simply exploring computer vision techniques, this project offers a practical example of what's possible with dedication and creativity.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
