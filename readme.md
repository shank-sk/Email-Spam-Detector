# 📧 Email Spam Detection using Machine Learning

## 🧠 Project Overview

The **Email Spam Detection System** is a machine learning project designed to classify emails as **spam** or **not spam (ham)**.  
It uses **Natural Language Processing (NLP)** techniques to preprocess text data and trains a **classification model** that accurately identifies spam messages based on their content.

---

## 🚀 Features

- 🧹 Text preprocessing (tokenization, stopword removal, stemming, lemmatization)
- 📊 Feature extraction using **Bag of Words (BoW)** / **TF-IDF**
- 🤖 Model training using algorithms like **Naive Bayes**, **Logistic Regression**, or **SVM**
- 📈 Model evaluation using accuracy, precision, recall, and F1-score
- 🔍 Real-time spam prediction on user input

---

## 🧩 Project Workflow

1. **Data Collection**

   - Dataset used: [SpamAssassin / Kaggle SMS Spam Collection](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

2. **Data Preprocessing**

   - Lowercasing, removing punctuation & special characters
   - Tokenization and stopword removal
   - Stemming and lemmatization

3. **Feature Extraction**

   - Convert text into numerical vectors using BoW or TF-IDF

4. **Model Building**

   - Train multiple classifiers (e.g., Naive Bayes, SVM, Logistic Regression)
   - Select the model with the best performance

5. **Evaluation**

   - Confusion Matrix, Accuracy, Precision, Recall, F1 Score

6. **Deployment (Optional)**
   - Integrate model into a web app using **Flask / Streamlit**

---

## 🧾 Technologies Used

- **Programming Language:** Python
- **Libraries & Frameworks:**
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `nltk`
  - `seaborn`
  - `streamlit` (if applicable)

---

## 📊 Example Output

| Email Text                                | Predicted Label |
| ----------------------------------------- | --------------- |
| "You have won a $1000 Walmart gift card!" | **Spam**        |
| "Meeting scheduled for 3 PM tomorrow."    | **Not Spam**    |

---

## 🧠 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 97.8% |
| Precision | 96.5% |
| Recall    | 95.9% |
| F1-Score  | 96.2% |

_(These values are sample results — update with your actual results.)_

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Email-Spam-Detection.git
cd Email-Spam-Detection
```

2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

3️⃣ Run the Notebook or Script

```bash
jupyter notebook
# or
python spam_detector.py
```

4️⃣ (Optional) Run Streamlit App

```bash
streamlit run app.py
```

## 🙌 Contributors

Built with ❤️ by [Shashank Arya](https://github.com/shank-sk)
