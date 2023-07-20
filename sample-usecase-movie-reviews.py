import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

""" 
Notes:
Set PATH: KMP_DUPLICATE_LIB_OK=TRUE
Positive sentiment (1)
Negative sentiment (0)
"""

# Movie reviews dataset
movie_reviews_data = {
    "reviews": [
        "This movie was fantastic! The plot was captivating and the acting was superb.",
        "I couldn't stand this movie. The story was boring and the acting was terrible.",
        "What a masterpiece! I was completely blown away by this film.",
        "This movie is a waste of time. Avoid it at all costs.",
        "The performances in this film were top-notch. I loved every moment of it.",
        "I regretted watching this movie. It was a total disappointment.",
        "I highly recommend this movie. It kept me on the edge of my seat.",
        "Awful movie. Don't bother watching it.",
        "A must-watch! This movie will stay with you long after it's over.",
        "I couldn't take my eyes off the screen. This movie is a true gem."
    ],
    "sentiments": [
        1,  
        0,  
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1
    ]
}

reviews = movie_reviews_data["reviews"]
sentiments = movie_reviews_data["sentiments"]
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(reviews).toarray()
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, sentiments, test_size=0.2, random_state=42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()  # Convert probabilities to binary labels (0 or 1)
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)