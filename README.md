# Train Yolo Model

This project explores training and using a YOLO (You Only Look Once) model for object detection, focusing on classifying facial expressions from a custom dataset. To create a more personalized dataset, I captured and labeled around 1500 images of my own facial expressions using Label Studio. It was more of a personal, self-tailored project rather than something meant for broader use, but it was a great way to learn and get familiar with YOLO in a practical setting.

## Project Overview

### 1. Custom Dataset & Manual Labeling

While many pre-labeled datasets are available online, I chose to take my own pictures to create a personalized dataset. I focused on classifying my own facial expressions—such as Natural, Happy, and Surprised—to add a personal touch to the project.  
- **Manual Labeling:** The manual process, though a bit time-consuming and requiring patience, felt refreshing and allowed me to make the project truly my own. It took a few hours to label all the pictures.
- **Label Studio:** I used Label Studio for labeling, which provided a flexible interface for annotating my images.

### 2. YOLO Model Training

The training process involves:
- Preparing the custom dataset with labeled images.
- Configuring the YOLO model with appropriate parameters.
- Training the model using the YOLO framework to detect and classify facial expressions.

### 3. Object Detection & Statistics Collection

After training, the model is used to:
- Detect face expressions in webcam, images and videos.
- Count the number of detections for each expression - in each session.
- Save these session statistics to a CSV file (`face_expression_stats.csv`) for further analysis.

### 4. Visualizing the Results

The project includes a visualization component:
- A script (`view_stats.py`) reads the CSV file and generates a pie chart that shows the distribution of expressions over all sessions.

## Project Structure

- **`yolo_detect.py`**: The main script for running object detection using the trained YOLO model. It loads the model, processes input sources (images, videos, or live camera streams), performs detection, counts and saves the detection statistics.
- **`view_stats.py`**: A script for visualizing the collected statistics. It reads the CSV file with detection statistics and generates visualizations to analyze the results.
- **`face_expression_stats.csv`**: A CSV file that stores the statistics of detected facial expressions.
- **`.gitignore`**: Specifies which files and directories to ignore in the Git repository.
- **`README.md`**: Providing an overview of the project and its usage.

## Deploying the Model on a PC

The easiest way to run Ultralytics YOLO models on a PC is using **Anaconda**, which helps set up a virtual Python environment with all necessary dependencies.

### **1. Download and Install Anaconda**
[Download Anaconda](https://anaconda.com/download) and follow the default installation settings.

### **2. Set up a virtual environment**
Open **Anaconda Prompt** (or a terminal on macOS/Linux) and create a new Python environment:
```bash
conda create --name yolo-env python=3.12 -y
conda activate yolo-env
```

Then, install Ultralytics:

```bash
pip install ultralytics
```

For NVIDIA GPU users, install the GPU-accelerated version of PyTorch:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Extract the trained model
Unzip ```my_model.zip``` and navigate into it's folder:
```bash
cd path/to/folder
```
#### 3.1. Download my model
If you want to, you may download my trained model and run it.

Simply click ```my_model.pt``` -> Download raw file.

### 4. Download and run yolo_detect.py
Download the detection script:

```bash
curl -o yolo_detect.py https://raw.githubusercontent.com/Etaizil/Train_Yolo_Model/refs/heads/main/yolo_detect.py
```

Run inference with a yolov8n model on a laptop or USB camera at 1280x720 resolution:

```bash
python yolo_detect.py --model my_model.pt --source laptop0 --resolution 720x720
```

This will open a window displaying a live feed with bounding boxes drawn around detected facial expressions.

Alternatively, to process an image, video, or folder of images, specify the source:

```bash
python yolo_detect.py --model my_model.pt --source path/to/image_or_video.mp4
```

## Inspiration & Future Improvements

This project was partly inspired by the ideas and techniques presented in this [YouTube video](https://www.youtube.com/watch?v=r0RspiLG260), which provided valuable guidance as I developed my workflow.  
While the project is currently a proof-of-concept, there are plenty of improvements that can be made:
- **Scaling Up:** Manual labeling of large datasets is time-consuming. Integrating semi-automated labeling tools could make this process more efficient.
- **Model Enhancements:** I only did a single training run for this project. Further tuning of the YOLO model and experimenting with different architectures could improve detection accuracy.
- **Extended Datasets:** Combining my custom dataset with pre-existing datasets might boost the model's performance and generalizability.

## Usage

To use the project, follow these steps:

1. **Train the YOLO Model:**
   - Prepare your dataset by capturing images and labeling them using Label Studio.
   - Configure and train the YOLO model using your custom dataset.

2. **Run Object Detection:**
   - Use the `yolo_detect.py` script to perform object detection on your images or videos. The script will display detection results (bounding boxes and labels) and save the detection statistics to `face_expression_stats.csv`. You are more than welcome to take the script and change it for your favor.
   - Feel free to modify the script to suit your needs.
3. **Visualize Statistics:**
   - Run the `view_stats.py` script to generate a visualization from the collected statistics. This will help you analyze the distribution of your facial expressions across sessions.

## Conclusion

This project is a personal, hands-on exploration of training a YOLO model for object detection, with a unique focus on classifying my own facial expressions. Although there is ample room for improvement - the manual approach provided a valuable learning experience and allowed me to tailor the dataset to my specific needs. Whether you're interested in custom model training or simply exploring computer vision techniques, this project offers a practical example of what's possible with dedication and creativity.


---
