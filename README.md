# Healthcare Assistant Chatbot

A sophisticated healthcare chatbot that combines neural networks and cosine similarity for natural language understanding. This hybrid approach offers robust and accurate responses to medical queries while maintaining conversation context.

## 🌟 Key Features

- **Hybrid Architecture**
  - Deep Neural Network with residual connections
  - TF-IDF vectorization with cosine similarity
  - Pattern matching and intent classification
  - Confidence-based response selection

- **Advanced UI**
  - Modern chat interface with bubble messages
  - Real-time response visualization
  - Interactive latent space visualization
  - Dark theme design

- **Medical Focus**
  - Specialized in healthcare domain
  - Response prioritization based on medical urgency
  - Structured conversation flow
  - Clear and concise medical information

## 🛠️ Technical Architecture

### 1. Text Processing
- TF-IDF vectorization
- N-gram range: (1, 2)
- Maximum features: 5000
- Cosine similarity threshold: 0.5

### 2. Neural Network
- Two dense layers (64 → 32 neurons)
- Strong L2 regularization (0.15)
- Batch normalization
- Dropout (0.5)
- Clipped gradients

### 3. Decision System
```python
if cosine_similarity >= 0.5:
    use_similarity_response()
elif neural_network_confidence > 0.7:
    use_neural_network_response()
elif cosine_similarity > 0.3:
    use_weak_similarity_response()
else:
    request_clarification()
```

## 📚 Dependencies

```python
tensorflow
numpy
pandas
plotly
umap-learn
scikit-learn
tkinter
```

## 🚀 Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/healthcare-chatbot.git
cd healthcare-chatbot
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Add your data file
- Create a JSON file named `data_healthV2.json`
- Follow this structure:
```json
{
    "intents": [
        {
            "tag": "symptom_fever",
            "patterns": [
                "I have a fever",
                "My temperature is high"
            ],
            "responses": [
                "How long have you had the fever?",
                "What's your temperature?"
            ]
        }
    ]
}
```

## 💻 Usage

### Running the GUI
```python
from chatbot import ChatbotInstance, launch_chatbot_gui

# Initialize chatbot
chatbot = ChatbotInstance()

# Launch GUI
launch_chatbot_gui(chatbot)
```

### Using the API
```python
# Get response for a query
response, confidence, tag, pattern, method = chatbot.get_best_response("I have a headache")

# Visualize latent space
visualize_embeddings_interactive("I have a headache")
```

## 📊 Performance

- Training accuracy: ~100%
- Validation accuracy: ~40% (significant given 136 classes)
- Response time: < 1 second
- Memory usage: ~200MB

## 🔄 Future Improvements

1. **Data Enhancement**
   - Increase training data volume
   - Add real patient-doctor conversations
   - Balance class distribution

2. **Architecture Updates**
   - Integrate pre-trained models (BERT, CamemBERT)
   - Add conversational context memory
   - Implement medical emergency detection

3. **Features**
   - Context tracking system
   - Response hierarchy based on severity
   - External medical source integration
   - Explanatory visualizations

## ⚠️ Disclaimer

This chatbot is a prototype and should not replace professional medical advice. Always consult healthcare professionals for medical decisions.

## 📝 License

MIT License - feel free to use and modify the code as needed.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## 📫 Contact

For questions and support, please open an issue in the GitHub repository.
