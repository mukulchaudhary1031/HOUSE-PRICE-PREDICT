House Price Prediction 🏠

A machine learning project to predict house prices based on property features using Python and Logistic Regression.
The app is deployed live on Render for instant access.


---

Table of Contents

Project Overview

Dataset

Technologies Used

Live Demo

Installation

Usage

Features

Future Scope

Author



---

Project Overview

This project predicts house prices based on property attributes.
It uses a Logistic Regression model for prediction and is deployed via FastAPI.
The project is structured for easy deployment and testing, though CI/CD pipelines are not integrated for now.


---

Dataset

The dataset contains property information with features like:

Number of Bedrooms

Number of Bathrooms

Square Footage

Location

Age of the Property

Other relevant features


Target: Price (Predicted house price)


---

Technologies Used

Python 🐍

Pandas & NumPy

Scikit-learn (Logistic Regression)

FastAPI (Backend)

Pickle (Model Serialization)

Docker (Optional, containerization ready)



---

Live Demo

Access the running application here:
https://house-price-predict-lrs0.onrender.com


---

Installation

1. Clone the repository:



git clone https://github.com/mukulchaudhary1031/<repo-name>.git
cd <repo-name>

2. Create virtual environment & install dependencies:



python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt

3. Run FastAPI server locally:



uvicorn main:app --reload

4. (Optional) Run with Docker:



docker build -t house-price-prediction .
docker run -p 8000:8000 house-price-prediction


---

Usage

1. Open browser at:



http://127.0.0.1:8000

2. Or use the live Render link for immediate access.


3. Enter property details in the form and submit.


4. Get predicted house price.


5. API testing is also possible via Postman or other HTTP clients.




---

Features

Preprocessing pipelines for numeric & categorical features

Handles missing data automatically

Logistic Regression model for prediction

Test accuracy & evaluation metrics

FastAPI backend for REST API

Dockerized deployment ready

Live deployment available on Render



---

Future Scope

Integrate more ML models (RandomForest, XGBoost, etc.) for better accuracy

Add visualizations like feature importance and price distribution

Add user authentication & property history tracking

Deploy on other cloud platforms (AWS/GCP)

Integrate CI/CD pipelines for automated deployment



---

Author

Mukul Chaudhary
GitHub: https://github.com/mukulchaudhary1031
FastAPI + ML enthusiast
Passionate about AI/ML full-stack applications
