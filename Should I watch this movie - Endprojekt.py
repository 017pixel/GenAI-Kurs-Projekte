from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Mehr Negative Reviews als Positive, Film wird nicht empfohlen
reviews = ['This movie was fantastic! A must-watch.', # die ganzen Reviews sind auf Englisch wegen Prozessing und so shit, vorsichtshalber 
           'I didn\'t enjoy this movie at all.',
           'Amazing storyline and great acting!',
           'The plot was dull and predictable.',
           'Loved the cinematography! Highly recommended.',
           'I found the movie boring and too long.',
           'What a wonderful experience! I would watch it again.',
           'Terrible acting and poor direction. Not worth it.',
           'An absolute masterpiece! I loved every moment.',
           'I regret watching this movie. It was a waste of time.',
           'The characters were well-developed and relatable.',
           'I couldn\'t connect with the characters at all.',
           'It was good!',
           'It was bad!',
           'It was very bad!!', 
           'I would not watch it!!',
           'Not worth it! Do not watch it!']

labels = ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'negative', 'negative', 'negative']

# Hier sind die Reviews umgedreht, dass das auch positiv sein kann und der Film empfohlen wird
"""" 
reviews = ['This movie was terrible! A waste of time.',
           'I absolutely loved this movie!',
           'Awful storyline and terrible acting!',
           'The plot was brilliant and unpredictable.',
           'Hated the cinematography! Not recommended.',
           'I found the movie engaging and perfectly paced.',
           'What a horrible experience! I would never watch it again.',
           'Excellent acting and great direction. Totally worth it.',
           'An absolute disaster! I hated every moment.',
           'I\'m so glad I watched this movie. It was time well spent.',
           'The characters were poorly-developed and unrelatable.',
           'I connected with the characters completely.',
           'It was bad!',
           'It was good!',
           'It was very good!!',
           'I would definitely watch it!!',
           'Worth it! You should watch it!']
labels = ['negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'positive', 'positive', 'positive']
"""

# In zahlen umwandeln
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(reviews)

# Aufteilen in Trainings- und Testdaten
x_train,  x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

#training hier 
model = MultinomialNB()
model.fit(x_train, y_train)

# Vorhersage
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Ausgabe der Genauigkeit
if accuracy > 0.8:
    print("Das Model sagt, dass der Film gut ist und du ihn gucken solltest.")
else:
    print("Das Model sagt, dass der Film schlecht ist und du ihn nicht gucken solltest.")