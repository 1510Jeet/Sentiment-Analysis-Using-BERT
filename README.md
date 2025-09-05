# 🧠 Twitter Sentiment Analysis using BERT

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive sentiment analysis project that classifies Twitter tweets into **positive**, **negative**, or **neutral** sentiments using a fine-tuned BERT model. The project includes data preprocessing, model training, and a user-friendly Streamlit web application for real-time sentiment prediction.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Project Overview

This project demonstrates the power of BERT (Bidirectional Encoder Representations from Transformers) for sentiment analysis on social media data. The model is fine-tuned on a Twitter dataset and achieves **75.6% accuracy** on the test set, significantly outperforming traditional machine learning approaches.

### Key Achievements
- ✅ **75.6% Test Accuracy** on sentiment classification
- ✅ **Advanced Text Preprocessing** with custom cleaning pipeline
- ✅ **Real-time Web Interface** using Streamlit
- ✅ **Production-ready Model** with proper serialization
- ✅ **Comprehensive Evaluation** with detailed metrics

## ✨ Features

### 🔧 Data Preprocessing
- **URL & HTML Removal**: Cleans tweets by removing web links and HTML tags
- **Special Character Handling**: Removes punctuation while preserving important text structure
- **Negation Processing**: Properly handles contractions and negations (e.g., "don't" → "do not")
- **Stopword Removal**: Eliminates common words that don't contribute to sentiment
- **Lemmatization**: Converts words to their base forms for better consistency
- **Duplicate Character Removal**: Handles repeated characters (e.g., "sooo" → "soo")
- **Case Normalization**: Converts all text to lowercase for uniformity

### 🤖 BERT Model Features
- **Fine-tuned BERT-base-uncased**: Leverages pre-trained BERT for transfer learning
- **Custom Configuration**: Optimized dropout rates (0.2) to prevent overfitting
- **Dynamic Padding**: Handles variable-length sequences efficiently
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Multi-class Classification**: Supports 3 sentiment categories

### 🌐 Web Application
- **Interactive Interface**: Clean, user-friendly Streamlit UI
- **Real-time Prediction**: Instant sentiment analysis
- **Batch Processing**: Supports single or multiple text inputs
- **Model Persistence**: Efficient model loading and caching

## 📊 Dataset

### Source
- **Dataset**: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **Size**: 27,481 training samples, 3,534 test samples
- **Format**: CSV with text and sentiment labels

### Dataset Structure
```
Columns:
├── textID: Unique identifier for each tweet
├── text: The actual tweet content
├── selected_text: Relevant portion of the tweet
├── sentiment: Ground truth label (positive/negative/neutral)
├── Time of Tweet: Timestamp information
├── Age of User: User demographic data
├── Country: Geographic information
├── Population -2020: Country population data
├── Land Area (Km²): Geographic metrics
└── Density (P/Km²): Population density
```

### Data Distribution
- **Positive**: ~32% of the dataset
- **Neutral**: ~40% of the dataset  
- **Negative**: ~28% of the dataset

## 🛠 Technology Stack

### Core Libraries
- **Python 3.8+**: Primary programming language
- **TensorFlow 2.x**: Deep learning framework
- **Transformers (Hugging Face)**: BERT model implementation
- **Streamlit**: Web application framework

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **NLTK**: Natural language processing
- **Scikit-learn**: Machine learning utilities

### Visualization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization

## 📁 Project Structure

```
Sentiment-Analysis-main/
├── App.py                              # Streamlit web application
├── fine-tune-bert-model.ipynb         # Jupyter notebook for model training
├── README.md                           # Project documentation
├── sentment analysis.pptx             # Project presentation
└── requirements.txt                   # Python dependencies (to be created)
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/1510Jeet/Sentiment-Analysis-Using-BERT.git
cd Sentiment-Analysis-Using-BERT
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv sentiment_env

# Activate virtual environment
# On Windows:
sentiment_env\Scripts\activate
# On macOS/Linux:
source sentiment_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install tensorflow==2.12.0
pip install transformers==4.21.0
pip install streamlit==1.28.0
pip install pandas==1.5.0
pip install numpy==1.24.0
pip install nltk==3.8.1
pip install scikit-learn==1.2.0
pip install matplotlib==3.6.0
pip install seaborn==0.12.0
```

### Step 4: Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 5: Download Dataset
1. Visit the [Kaggle Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
2. Download the dataset files
3. Place them in a `data/` folder in your project directory

### Step 6: Train the Model
```bash
# Run the Jupyter notebook
jupyter notebook fine-tune-bert-model.ipynb
```

**Note**: Update the file paths in the notebook to match your local setup.

### Step 7: Update Model Paths
Before running the Streamlit app, update the model path in `App.py`:

```python
# Change this line to your actual model path
path = r'C:\Users\Tarek Hesham\Sent_model'  # Update this path
```

## 💻 Usage

### Running the Web Application
```bash
streamlit run App.py
```

The application will be available at `http://localhost:8501`

### Using the Application
1. **Open your browser** and navigate to the Streamlit app
2. **Enter text** in the text area (single tweet or multiple tweets)
3. **Click "Predict Sentiment"** to get the analysis
4. **View results** showing the predicted sentiment (positive/negative/neutral)

### Example Usage
```python
# Example input
"I love this new product! It's amazing!"

# Expected output
Predicted Sentiment: ['positive']
```

## 📈 Model Performance

### Test Results
- **Test Accuracy**: 75.6%
- **Test Loss**: 0.583

### Detailed Metrics
```
Classification Report:
               precision    recall  f1-score   support

    negative       0.71      0.82      0.76      1001
     neutral       0.75      0.66      0.71      1430
    positive       0.80      0.82      0.81      1103

    accuracy                           0.76      3534
   macro avg       0.76      0.77      0.76      3534
weighted avg       0.76      0.76      0.75      3534
```

### Training Progress
- **Epochs**: 6 (with early stopping)
- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Optimizer**: Adam
- **Validation Split**: 20%

## 🔧 API Reference

### Main Functions

#### `Get_sentiment(Review, Tokenizer, Model)`
Predicts sentiment for given text input.

**Parameters:**
- `Review` (str or list): Text to analyze
- `Tokenizer` (BertTokenizer): BERT tokenizer instance
- `Model` (TFBertForSequenceClassification): Trained BERT model

**Returns:**
- `list`: Predicted sentiment labels

**Example:**
```python
result = Get_sentiment("I love this product!")
print(result)  # ['positive']
```

#### `text_preprocessing` Class
Handles comprehensive text cleaning and preprocessing.

**Methods:**
- `clean(text)`: Main preprocessing function
- `__init__(stemming, Lemmatisation)`: Initialize with processing options

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Project Maintainer**: Jeet
- **GitHub**: [@1510Jeet](https://github.com/1510Jeet)
- **Repository**: [Sentiment-Analysis-Using-BERT](https://github.com/1510Jeet/Sentiment-Analysis-Using-BERT)

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and BERT implementation
- **Kaggle** for providing the Twitter sentiment dataset
- **Streamlit** for the excellent web application framework
- **TensorFlow** team for the robust deep learning framework

## 📚 Additional Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)

---

⭐ **Star this repository if you found it helpful!**
