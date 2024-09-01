import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot

# Load the data for search recommendation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Define the search queries and recommended search terms
search_queries = [
    "best pizza in town",
    "vegetarian restaurants near me",
    "how to train a dog",
    "top 10 action movies",
    "healthy breakfast ideas",
    "fun things to do on a rainy day",
    "how to learn a new language",
    "best hiking trails near me",
    "how to start a small business",
    "DIY home decor ideas",
    "best beaches in the world",
    "how to make pizza dough",
    "top 5 tourist attractions in Paris",
    "best budget smartphones",
    "how to learn programming"
]

recommended_terms = [
    "pizza",
    "vegetarian",
    "dog training",
    "action movies",
    "breakfast",
    "rainy day activities",
    "language learning",
    "hiking trails",
    "small business",
    "home decor",
    "beaches",
    "pizza dough",
    "tourist attractions",
    "budget smartphones",
    "programming tutorials"
]

# Create a CountVectorizer object with a vocabulary of 1000 words
vectorizer = CountVectorizer(max_features=1000, binary=True)

# Convert the search queries and recommended terms to matrices of one-hot encoded vectors
X = vectorizer.fit_transform(search_queries).toarray()
y = vectorizer.fit_transform(recommended_terms).toarray()

# Use the matrices of one-hot encoded vectors to create X_train and y_train inputs for the MLP program
X_train = np.array(X)
y_train = np.array(y)


# Define the MLP model architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


model = MLP(input_size=X_train.shape[1], hidden_size1=128,
            hidden_size2=64, output_size=y_train.shape[1])

# Visualize the MLP model with Graphviz
x = torch.randn(1, X_train.shape[1])
make_dot(model(x), params=dict(model.named_parameters()))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 100
batch_size = 128
num_batches = len(X_train) // batch_size

if num_batches == 0:
    num_batches = 1

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        inputs = torch.FloatTensor(X_train[start:end])
        labels = torch.LongTensor(y_train[start:end])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Compute the accuracy on the training set
    with torch.no_grad():
        inputs = torch.FloatTensor(X_train)
        labels = torch.LongTensor(y_train)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        epoch_acc = (predicted == torch.max(labels, 1)
                     [1]).sum().item() / len(X_train)

    # Print the loss and accuracy for the current epoch
    print(
        f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}, Accuracy: {epoch_acc:.4f}')
