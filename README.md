# 🎬 Movie Review Sentiment Analyzer

A lightweight Flask web application that predicts the **sentiment** of a movie review as **positive** or **negative** using a **Naive Bayes machine learning model**.

---

## Features:

- User-friendly web interface to input movie reviews.
- Uses a trained machine learning model (`model.pkl`) for sentiment prediction.
- Built with Flask, scikit-learn, HTML, and JavaScript.
- Clean and minimal UI, ideal for small projects or demonstrations.

---

## Technologies Used:

- **Python 3**
- **Flask** – Web framework
- **scikit-learn** – Model training
- **pandas** – Data handling
- **nltk** – Text processing (stopwords)
- **pickle** – Model serialization
- **HTML/CSS/JavaScript** – Frontend


---

## 🛠️ Installation and Running the App

Follow these simple steps to run the project locally:

### Step 1: Clone the Repository

```bash
git clone https://github.com/VaishnaviVadla33/SentimentAnalysis_FlaskApplication.git
cd SentimentAnalysis_FlaskApplication
```

### Step 2: Create a Virtual Environment
```bash
python -m venv myenv
```

### Step 3: Activate the Environment
#### On Windows:
```bash
myenv\Scripts\activate
```
#### On macOS/Linux:
```bash
source myenv/bin/activate
```

### Step 4: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 5: Run the Application
```bash
python app.py
```

Then open your browser and go to:
http://127.0.0.1:5000




