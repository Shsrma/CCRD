# Credit Card Fraud Detection System

A comprehensive machine learning-powered system for detecting fraudulent credit card transactions in real-time.

## 🚀 Features

- **Advanced ML Models**: Logistic Regression, Random Forest, XGBoost, SVM, and Naive Bayes
- **Class Imbalance Handling**: SMOTE, ADASYN, and class weight balancing
- **Comprehensive Evaluation**: ROC-AUC, Precision-Recall, F1-score, and more
- **Production Ready**: FastAPI backend with authentication, logging, and monitoring
- **Reproducible Training**: Configurable experiments with full reproducibility
- **Real-time API**: RESTful API for fraud prediction with authentication
- **Web Dashboard**: Frontend for monitoring fraud alerts and system settings

## 🏗️ Architecture

```
CCRD/
├── backend/
│   ├── app/
│   │   ├── api/           # API routes and controllers
│   │   ├── core/          # Configuration and logging
│   │   ├── database/      # Database models and connections
│   │   ├── ml/            # ML modules
│   │   │   ├── preprocessing.py   # Data preprocessing
│   │   │   ├── models.py          # ML models
│   │   │   ├── evaluation.py      # Model evaluation
│   │   │   ├── imbalancer.py      # Imbalance handling
│   │   │   └── trainer.py         # Model training
│   │   └── main.py        # Main application
│   ├── ml/
│   │   └── train_model.py # Training script
│   └── requirements.txt
├── frontend/
│   ├── index.html         # Main dashboard
│   ├── alerts.html       # Fraud alerts view
│   ├── login.html        # Authentication
│   └── style.css         # Styling
└── data/                 # Dataset (not included in repo)
    └── creditcard.csv
```

## 📋 Prerequisites

- Python 3.8+
- Node.js (for frontend, optional)
- Credit card fraud dataset (download from Kaggle)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CCRD.git
cd CCRD
```

2. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Download the credit card fraud dataset:
```bash
# Download creditcard.csv from Kaggle and place in data/ directory
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
mkdir -p data
# Place creditcard.csv in the data directory
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your own values
```

## 🚀 Usage

### 1. Train the Model

```bash
cd backend
python -m ml.train_model --model random_forest --imbalance-method class_weights
```

Available models: `logistic_regression`, `random_forest`, `xgboost`, `naive_bayes`, `svm`
Available imbalance methods: `class_weights`, `smote`, `adasyn`, `borderline_smote`, `svm_smote`

### 2. Start the API Server

```bash
cd backend
python -m app.main
```

The API will be available at `http://localhost:8000`

### 3. Access the Web Interface

Open `frontend/index.html` in your browser or access the API documentation at `http://localhost:8000/docs`

## 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.9992 | 0.9310 | 0.7241 | 0.8148 | 0.9912 |
| XGBoost | 0.9991 | 0.9286 | 0.7143 | 0.8070 | 0.9908 |
| Logistic Reg. | 0.9990 | 0.9130 | 0.7000 | 0.7925 | 0.9895 |

## 🔐 API Endpoints

- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/auth/signup` - User registration
- `POST /predict` - Fraud prediction (requires authentication)
- `GET /alerts` - Get fraud alerts (requires authentication)
- `GET /health` - Health check

## 📈 Training Configuration

The system supports configurable training parameters:

```bash
python -m ml.train_model \
  --model xgboost \
  --random-state 42 \
  --test-size 0.2 \
  --imbalance-method smote \
  --save-model
```

## 🧪 Evaluation Metrics

The system calculates comprehensive metrics:
- **Accuracy**: Overall correctness
- **Precision**: True positives among predicted positives
- **Recall/Sensitivity**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Precision-Recall AUC**: Area under the PR curve
- **Matthews Correlation Coefficient**: Quality of binary classifications
- **Cohen's Kappa**: Agreement beyond chance

## 🏷️ Key Innovations

1. **Advanced Preprocessing**: Robust data validation and feature scaling
2. **Multiple Imbalance Techniques**: Various approaches to handle class imbalance
3. **Model Comparison**: Built-in capability to compare multiple models
4. **Cross-Validation**: K-fold validation for robust evaluation
5. **Reproducible Experiments**: Full control over randomness for reproducibility
6. **Production Monitoring**: Structured logging and error handling
7. **Security**: JWT-based authentication and secure API endpoints

## 🎯 Business Impact

- **Reduced False Positives**: Better precision reduces customer friction
- **Improved Detection**: Higher recall catches more fraudulent transactions
- **Scalable Architecture**: Designed for high-volume transaction processing
- **Real-time Response**: Fast inference for live transaction screening

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support, please open an issue in the repository or contact the maintainers.

---

Made with ❤️ for the open-source community