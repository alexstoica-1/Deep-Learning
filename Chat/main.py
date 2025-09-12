import os
import json
import random

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Optional, Tuple


class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings = None):
        self.model: Optional[ChatbotModel] = None
        self.intents_path: str = intents_path

        self.documents: List[Tuple[List[str], str]] = []
        self.vocabulary: List[str] = []
        self.intents: List[str] = []
        self.intents_responses: Dict[str, List[str]] = {}

        self.function_mappings = function_mappings

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Intents file not found: {self.intents_path}")

        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)

        for intent in intents_data.get('intents', []):
            if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent.get('responses', [])

            for pattern in intent.get('patterns', []):
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, intent['tag']))

        # De-duplicate/sort vocabulary once after processing all intents
        self.vocabulary = sorted(set(self.vocabulary))

        if not self.documents:
            raise ValueError("No training patterns found in intents file. Add patterns under each intent to train the model.")

    def prepare_data(self):
        bags = []
        indices = []

        if not self.documents:
            raise ValueError("No documents to prepare. Did you run parse_intents() and does intents.json contain patterns?")

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        if self.X is None or self.y is None:
            raise ValueError("Training data not prepared. Call prepare_data() before train_model().")
        if self.X.size == 0 or self.y.size == 0:
            raise ValueError("Training data is empty. Ensure intents.json contains patterns.")
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_size = int(self.X.shape[1])  # type: ignore[union-attr]
        self.model = ChatbotModel(input_size, len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        if self.X is None:
            raise ValueError("Input dimensions unknown. Prepare data before saving dimensions.")
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': int(self.X.shape[1]), 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    def process_message(self, input_message):
        if self.model is None:
            raise ValueError("Model is not loaded or trained. Call train_model() or load_model() first.")
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        if not self.intents:
            raise ValueError("No intents available. Did you parse intents?")

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[int(predicted_class_index)]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None


def get_stocks():
    stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']

    print(random.sample(stocks, 3))


if __name__ == '__main__':
    assistant = ChatbotAssistant('/Users/alexisgod/Desktop/Deep Learning/Chat/intents.json', function_mappings = {'stocks': get_stocks})
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)

    assistant.save_model('/Users/alexisgod/Desktop/Deep Learning/Chat/chatbot_model.pth', '/Users/alexisgod/Desktop/Deep Learning/Chat/dimensions.json')

    assistant = ChatbotAssistant('/Users/alexisgod/Desktop/Deep Learning/Chat/intents.json', function_mappings = {'stocks': get_stocks})
    assistant.parse_intents()
    assistant.load_model('/Users/alexisgod/Desktop/Deep Learning/Chat/chatbot_model.pth', '/Users/alexisgod/Desktop/Deep Learning/Chat/dimensions.json')

    while True:
        message = input('Enter your message:')

        if message == '/quit':
            break

        print(assistant.process_message(message))