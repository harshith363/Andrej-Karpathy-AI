import numpy as np
import pandas as pd
import tensorflow as tf

# Load data
data = pd.read_csv('search_data.csv')
num_users = len(data['user_id'].unique())
num_items = len(data['item_id'].unique())

# Split data into training and test sets
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Convert data into user-item matrix
train_matrix = np.zeros((num_users, num_items))
for row in train_data.itertuples():
    train_matrix[row[1]-1, row[2]-1] = row[3]

# Define model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(num_items,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_items, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train model
history = model.fit(train_matrix, train_matrix, epochs=10, batch_size=64)

# Evaluate model on test set
test_matrix = np.zeros((num_users, num_items))
for row in test_data.itertuples():
    test_matrix[row[1]-1, row[2]-1] = row[3]
predictions = model.predict(test_matrix)
test_loss = model.evaluate(test_matrix, test_matrix)

# Recommend items for a given user


def recommend_items(user_id, top_n=10):
    user_matrix = np.zeros((1, num_items))
    user_matrix[0] = train_matrix[user_id-1]
    predictions = model.predict(user_matrix)
    top_items = np.argsort(predictions)[0][::-1][:top_n]
    return top_items + 1


# Example usage
recommended_items = recommend_items(1)
print(recommended_items)
