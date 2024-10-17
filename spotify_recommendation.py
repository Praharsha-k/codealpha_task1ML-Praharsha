# Importing necessary libraries
import pandas as pd  # For working with the dataset
import numpy as np  # For working with numbers and arrays
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # For normalizing the data
from sklearn.ensemble import RandomForestClassifier  # For making predictions
from sklearn.metrics import accuracy_score, classification_report  # For evaluating the predictions

# Load the tracks.csv file (located in the same folder as the Python script)
df = pd.read_csv('C:/Users/praharsha k/OneDrive/Pictures/spot/tracks.csv')

# Display the first few rows to understand the data
print(df.head())

# Selecting relevant features from tracks.csv (adjust the columns based on your dataset)
# You can modify these based on the columns available in the dataset
X = df[['danceability', 'energy', 'acousticness', 'speechiness', 'instrumentalness']]

# Creating a random 'repeated' column for now (1 = user will replay the song, 0 = user will not)
# In practice, this column would be based on real data
df['repeated'] = np.random.choice([0, 1], size=len(df))

# The target variable 'y' is whether the song is replayed
y = df['repeated']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data to make it easier for the model to learn
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training data
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Check the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print a detailed classification report
print(classification_report(y_test, y_pred))

# Example of predicting for a new song (values based on the features in tracks.csv)
new_song = pd.DataFrame({
    'danceability': [0.6],
    'energy': [0.8],
    'acousticness': [0.1],
    'speechiness': [0.05],
    'instrumentalness': [0.0]
})

# Standardize the new song's features using the same scaler
new_song_scaled = scaler.transform(new_song)

# Predict whether the user will replay the song
prediction = model.predict(new_song_scaled)

# Display the prediction result
if prediction == 1:
    print("The user will probably listen to this song again.")
else:
    print("The user probably won't listen to this song again.")
