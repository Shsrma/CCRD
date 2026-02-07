#!/bin/bash
echo "========================================"
echo "Credit Card Fraud Detection - Setup Guide"
echo "========================================"

echo ""
echo "1. Installing Dependencies..."
cd backend
pip install -r requirements.txt

echo ""
echo "2. Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
fi

echo ""
echo "3. Training the ML Model (Random Forest with class weights)..."
echo "   Available models: logistic_regression, random_forest, xgboost, naive_bayes, svm"
echo "   Available imbalance methods: class_weights, smote, adasyn, borderline_smote, svm_smote"
python -m ml.train_model --model random_forest --imbalance-method class_weights

echo ""
echo "4. Starting the API Server..."
echo "   The API will be available at http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "To start the server manually, run: python -m app.main"