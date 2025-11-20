# Safety Helmet Detection

### **How to use this:**

1.  Open `README.md` in VS Code.
2.  Delete everything currently in it.
3.  **Copy and paste** the block below.
4.  Save, Commit, and Push.

-----

````markdown
# 👷 Safety Helmet Detection

This project is an automated computer vision system designed to detect safety helmets in videos and images. It utilizes a custom-trained **YOLOv5** model to ensure safety compliance in construction and industrial environments.

## 📂 Project Structure
* **`yolov5/`**: Contains the core detection algorithms and source code.
* **`vit_final/`**: Stores the trained model weights (`best.pt`) and training performance graphs (confusion matrices, F1 curves).
* **`test_videos/`**: Directory for input videos to test the detection.

---

## 🛠️ Installation & Setup

Follow these steps to set up the project on your local machine (Mac/Linux/Windows).

### 1. Clone the Repository
```bash
git clone [https://github.com/sehrishniazi/Safety_helmet_detection.git](https://github.com/sehrishniazi/Safety_helmet_detection.git)
cd Safety_helmet_detection
````

### 2\. Create a Virtual Environment

It is recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate
```

### 3\. Install Dependencies

Ensure you are inside the `yolov5` directory or have the requirements file ready.

```bash
pip install -r requirements.txt
```

-----

## 🚀 How to Run Detection

To detect safety helmets in your test videos, use the following command.

**Note:** Ensure you are inside the `yolov5` directory before running the script (or adjust the path to `detect.py`).

```bash
cd yolov5
python3 detect.py --weights ../vit_final/weights/best.pt --img 320 --conf 0.25 --source "../test_videos"
```

### Command Arguments Explained:

  * `--weights`: Path to your trained model (`vit_final/weights/best.pt`).
  * `--img 320`: Resizes input to 320 pixels for faster processing.
  * `--conf 0.25`: Sets the confidence threshold to 25%.
  * `--source`: Path to the folder containing images or videos to test.

-----

## 📊 Training Results

The model training performance is documented in the **`vit_final`** folder. Key metrics include:

  * **Confusion Matrix**: `vit_final/confusion_matrix.png`
  * **F1 Score Curve**: `vit_final/F1_curve.png`
  * **Precision-Recall**: `vit_final/PR_curve.png`

-----

## 🤝 Credits & Acknowledgments

  * **Original Repository**: [M-Haris7/Safety\_helmet\_detection](https://github.com/M-Haris7/Safety_helmet_detection)
  * **Base Architecture**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)

<!-- end list -->

```
```
