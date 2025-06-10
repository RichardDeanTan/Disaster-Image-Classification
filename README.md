# Disaster Image Classification

This project utilizes a deep learning model to classify images of disasters into four categories: **Earthquake**, **Land Slide**, **Urban Fire**, and **Water Disaster**. The application is built with a FastAPI backend to serve the model and a Streamlit frontend for interactive predictions and model comparison.

## 💡 Features
-   **Deep Learning Model**: Uses a powerful `EfficientNetB2` model fine-tuned for high accuracy on disaster images.
-   **Model Comparison**: Interactively compare the performance of the model before and after fine-tuning on your own images.
-   **Interactive UI**: A user-friendly web interface built with Streamlit allows for easy image uploads and sample testing.
-   **RESTful API**: A robust backend powered by FastAPI serves the model predictions, making the system modular and scalable.
-   **Two Deployment Modes**: The Streamlit app can run in two modes:
    1.  **Direct Integration**: Loads the models directly for a self-contained application.
    2.  **FastAPI Server**: Communicates with the backend API, demonstrating a full-stack architecture.

## 📂 Project Structure

```
.
├── backend/
│   ├── models/
│   │   ├── effnet_before.keras  # Base EfficientNet model
│   │   └── effnet_tune.keras    # Fine-tuned EfficientNet model
│   └── main.py                  # FastAPI application script
├── frontend/
│   ├── samples/                 # Sample images for each category
│   │   ├── Earthquake.jpg
│   │   ├── Land_Slide.jpg
│   │   ├── Urban_Fire.jpg
│   │   └── Water_Disaster.jpg
│   ├── app.py                   # Main Streamlit application
│   ├── index.html               # (Alternative) Basic HTML frontend
│   ├── style.css                # (Alternative) Styles for the HTML page
│   └── script.js                # (Alternative) JavaScript for the HTML page
├── DisasterClassification_EfficientNetB2.ipynb # Jupyter Notebook for model training
├── requirements.txt             # Python dependencies for the project
└── README.md                    # Readme File
```

## 🚀 How to Run the Project Locally
This project has two main components: the **FastAPI backend** and the **Streamlit frontend**. You need to run both simultaneously in separate terminals for the full experience.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/RichardDeanTan/Disaster-Image-Classification.git](https://github.com/RichardDeanTan/Disaster-Image-Classification.git)
cd your-repo-name
```

### Step 2: Install Dependencies
Install all the necessary Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 3: Run the Backend Server
In your first terminal, navigate to the project root and start the FastAPI server using Uvicorn.

```bash
uvicorn backend.main:app --reload
```

The API server should now be running at `http://127.0.0.1:8000`.

### Step 4: Run the Streamlit Frontend
Open a second terminal. Navigate to the project root and run the Streamlit app.

```bash
streamlit run frontend/app.py
```

Your web browser should automatically open with the interactive application. In the app's sidebar, you can choose the "FastAPI Server" mode to connect to your local backend.

---

### Alternative HTML/CSS/JS Frontend
This repository also includes a basic web app built with HTML, CSS, and JavaScript located in the `frontend/` directory. This version is provided as a simple example of how a traditional web app could consume the FastAPI backend.

To use it, you must first have the backend server running (Step 3). Then, you can simply open the `frontend/index.html` file in your web browser. Note that for the JavaScript `fetch` calls to work correctly, you might need to serve the files using a simple local server (like Python's `http.server` or the Live Server extension in VS Code) to avoid CORS issues.

## 🚀 Run App from Streamlit
Click the link below to run streamlit app without cloning repository:
#### [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://richarddeantanjaya-disaster-image-classification.streamlit.app/)

## 🧠 Model Details
-   **Base Model**: `effnet_before.keras` is the pre-trained EfficientNetB2 model from Keras, without any fine-tuning on the disaster dataset.
-   **Tuned Model**: `effnet_tune.keras` is the result of fine-tuning the base model on our specific disaster image dataset, leading to significantly better performance.
-   **Training Notebook**: The entire process of data preprocessing, model creation, training, and evaluation is documented in the `DisasterClassification_EfficientNetB2.ipynb` Jupyter Notebook.

## 📦 Tech Stack
-   **Backend**: Python, FastAPI, Uvicorn
-   **Frontend**: Streamlit, Plotly
-   **ML/Data Science**: TensorFlow, Keras, NumPy, Pillow

## 📝 License
This project is open-source and available for anyone to use.