# Notwendige Bibliotheken importieren
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Erweiterte Trainingsdaten mit mehr Tech-Begriffen
tech_texts = [
    # Programmierung
    'Ich liebe Programmieren.',
    'Python ist fantastisch.',
    'Ich programmiere gerne.',
    'Programmieren macht Spaß.',
    'Ich lerne programmieren.',
    'JavaScript ist vielseitig.',
    'HTML und CSS sind wichtig.',
    'Ich programmiere in Java.',
    'C++ ist eine mächtige Sprache.',
    'Ich entwickle Software.',
    'Code schreiben ist kreativ.',
    'Debugging ist wichtig.',
    'Ich liebe es zu coden.',
    'Webentwicklung ist vielseitig.',
    'Mobile Apps entwickeln ist kreativ.',
    'Die Programmierung ist eine Kunst.',
    'Die Entwicklung von Algorithmen ist anspruchsvoll.',
    'Softwareentwicklung ist meine Leidenschaft.',
    'Die Programmierung erfordert Logik.',
    'API-Design ist eine Kunst.',
    'Ich entwickle Webanwendungen.',
    'Frontend-Entwicklung ist spannend.',
    'Backend-Systeme sind komplex.',
    'Ich arbeite mit Frameworks.',
    'Version Control ist essentiell.',
    
    # KI und Machine Learning
    'Ich mag maschinelles Lernen.',
    'Maschinelles Lernen ist faszinierend.',
    'Künstliche Intelligenz verändert die Welt.',
    'Natürliche Sprachverarbeitung ist ein Teil von KI.',
    'Künstliche Intelligenz ist überall.',
    'Deep Learning ist ein Teilbereich der KI.',
    'Die künstliche Intelligenz lernt.',
    'Künstliche neuronale Netze sind spannend.',
    'Maschinelles Sehen ist komplex.',
    'Machine Learning Algorithmen sind mächtig.',
    'Neuronale Netze sind faszinierend.',
    'KI revolutioniert die Industrie.',
    'Ich trainiere ML-Modelle.',
    'Supervised Learning ist effektiv.',
    'Unsupervised Learning entdeckt Muster.',
    'Reinforcement Learning ist spannend.',
    'Computer Vision ist beeindruckend.',
    'NLP versteht menschliche Sprache.',
    'Ich arbeite mit TensorFlow.',
    'PyTorch ist ein tolles Framework.',
    
    # Daten und Analyse
    'Ich mag Algorithmen.',
    'Datenbanken sind komplex.',
    'Big Data ist eine Herausforderung.',
    'Die Datenanalyse ist entscheidend.',
    'Big Data Analytics ist mächtig.',
    'Die Datenbank ist optimiert.',
    'Die Datenwissenschaft ist zukunftsträchtig.',
    'SQL ist für Datenbanken wichtig.',
    'NoSQL bietet Flexibilität.',
    'Data Mining entdeckt Erkenntnisse.',
    'Datenvisualisierung ist wichtig.',
    'ETL-Prozesse verarbeiten Daten.',
    'Data Warehousing speichert Informationen.',
    'Analytics Dashboard zeigen Trends.',
    'Ich analysiere große Datenmengen.',
    'Statistik hilft bei der Datenanalyse.',
    'Predictive Analytics ist wertvoll.',
    'Business Intelligence informiert Entscheidungen.',
    'Ich erstelle Datenmodelle.',
    'Data Science kombiniert verschiedene Bereiche.',
    
    # Technologie und Systeme
    'Cloud Computing wird immer wichtiger.',
    'Cybersecurity ist entscheidend.',
    'Blockchain-Technologie ist innovativ.',
    'Quantencomputing ist die Zukunft.',
    'Netzwerksicherheit ist unerlässlich.',
    'Virtuelle Realität ist immersiv.',
    'DevOps verbessert die Zusammenarbeit.',
    'Augmented Reality bietet neue Möglichkeiten.',
    'Die IT-Infrastruktur ist das Rückgrat.',
    'Ich mag Kryptographie.',
    'Ich liebe die Technik.',
    'Das Internet ist riesig.',
    'Die Softwarearchitektur ist wichtig.',
    'Die Cloud ist skalierbar.',
    'Die Webseiten sind responsiv.',
    'Die Cybersicherheit ist eine Priorität.',
    'Die Blockchain ist dezentral.',
    'Das Quantencomputing ist revolutionär.',
    'Die Netzwerke sind sicher.',
    'Die mobile Entwicklung ist schnelllebig.',
    'Die Augmented Reality ist interaktiv.',
    'Das IT-System ist stabil.',
    'Die Deep-Learning-Modelle sind komplex.',
    'Die Kryptowährungen sind volatil.',
    'Docker Container sind praktisch.',
    'Serverlose Architektur ist effizient.',
    'Microservices sind skalierbar.',
    'Cloud-native Anwendungen sind modern.',
    'Container-Orchestrierung ist komplex.',
    'Kubernetes verwaltet Container.',
    'AWS bietet viele Services.',
    'Azure ist Microsofts Cloud-Plattform.',
    'Google Cloud ist innovativ.',
    'Ich arbeite mit REST APIs.',
    'GraphQL ist eine Alternative zu REST.',
    'Serverless Computing ist effizient.',
    'Edge Computing ist dezentral.',
    'IoT verbindet Geräte.',
    'Blockchain bietet Transparenz.',
    'Smart Contracts automatisieren Prozesse.',
    'Ich konfiguriere Server.',
    'Linux ist ein stabiles Betriebssystem.',
    'Open Source fördert Innovation.',
    'Git verwaltet Code-Versionen.',
    'CI/CD automatisiert Deployment.',
    'Monitoring überwacht Systeme.',
    'Load Balancing verteilt Traffic.',
    'Caching verbessert Performance.',
    'Datenbank-Optimierung ist wichtig.',
    'Security Audits sind notwendig.',
    
    # Hardware und Computer
    'Computer sind faszinierend.',
    'Prozessoren werden immer schneller.',
    'GPUs beschleunigen Berechnungen.',
    'SSDs sind schneller als HDDs.',
    'RAM bestimmt die Geschwindigkeit.',
    'Motherboards verbinden Komponenten.',
    'Gaming-PCs sind leistungsstark.',
    'Server verarbeiten Anfragen.',
    'Laptops sind mobil.',
    'Tablets sind praktisch.',
    'Smartphones sind Computer.',
    'Ich baue PCs zusammen.',
    'Hardware-Upgrades verbessern Leistung.',
    'Overclocking erhöht die Geschwindigkeit.',
    'Kühlung ist wichtig für Computer.',
    'Netzteil versorgt Komponenten.',
    'BIOS konfiguriert Hardware.',
    'Treiber ermöglichen Hardware-Kommunikation.',
    'USB verbindet Peripheriegeräte.',
    'HDMI überträgt Video und Audio.'
]

non_tech_texts = [
    # Alltag und persönliche Aktivitäten
    'Das Wetter ist heute schön.',
    'Der Himmel ist blau.',
    'Kochen macht Spaß.',
    'Die Sonne scheint hell.',
    'Ich lese gerne Bücher.',
    'Ich gehe gerne spazieren.',
    'Der Kaffee schmeckt gut.',
    'Ich höre Musik zum Entspannen.',
    'Mein Hund spielt im Garten.',
    'Ich trinke gerne Tee.',
    'Die Blumen blühen im Frühling.',
    'Ich schaue gerne Filme.',
    'Der Vogel singt ein Lied.',
    'Ich backe gerne Kuchen.',
    'Die Berge sind beeindruckend.',
    'Ich gehe gerne schwimmen.',
    'Der Regen prasselt ans Fenster.',
    'Ich besuche gerne Museen.',
    'Die Vögel zwitschern.',
    'Ich male gerne Bilder.',
    'Der Wind weht sanft.',
    'Ich spiele gerne Brettspiele.',
    'Ich genieße die Ruhe.',
    'Der Mond leuchtet hell.',
    'Ich fahre gerne Fahrrad.',
    'Die Wolken ziehen vorbei.',
    'Ich gehe gerne ins Kino.',
    'Ich räume mein Zimmer auf.',
    'Ich mache Sport.',
    'Ich koche heute Abend.',
    'Ich pflanze Blumen im Garten.',
    'Ich gehe einkaufen.',
    'Ich lese eine Zeitung.',
    'Ich gehe ins Fitnessstudio.',
    'Ich spiele Gitarre.',
    'Ich höre ein Hörbuch.',
    'Ich gehe mit meinem Hund spazieren.',
    'Ich schlafe gerne aus.',
    'Ich trinke ein Glas Wasser.',
    'Ich gehe heute früh ins Bett.',
    'Ich besuche Freunde.',
    'Ich gehe am Strand spazieren.',
    'Ich sehe fern.',
    'Ich mache ein Nickerchen.',
    'Ich esse ein Stück Obst.',
    'Ich putze meine Zähne.',
    'Ich mache einen Spaziergang im Park.',
    'Ich trinke meinen Morgenkaffee.',
    'Die Katze schläft auf dem Sofa.',
    'Ich gehe heute schwimmen.',
    'Der Garten braucht Wasser.',
    'Ich höre Podcasts beim Joggen.',
    
    # Familie und Beziehungen
    'Meine Familie ist wichtig.',
    'Ich liebe meine Kinder.',
    'Freundschaft ist wertvoll.',
    'Ich vermisse meine Eltern.',
    'Mein Partner macht mich glücklich.',
    'Wir feiern heute Geburtstag.',
    'Die Hochzeit war wunderschön.',
    'Ich besuche meine Großeltern.',
    'Kinder spielen im Park.',
    'Wir machen einen Familienausflug.',
    'Ich rufe meine Schwester an.',
    'Mein Bruder hilft mir.',
    'Wir essen zusammen Abend.',
    'Die Baby lacht so süß.',
    'Ich gehe mit Papa spazieren.',
    'Mama kocht mein Lieblingsessen.',
    'Wir schauen gemeinsam Filme.',
    'Ich spiele mit den Kindern.',
    'Oma erzählt Geschichten.',
    'Opa repariert das Fahrrad.',
    
    # Natur und Umwelt
    'Die Natur ist wunderschön.',
    'Bäume spenden Schatten.',
    'Vögel singen am Morgen.',
    'Der Wald ist friedlich.',
    'Blumen duften herrlich.',
    'Der See ist kristallklar.',
    'Berge sind majestätisch.',
    'Der Sonnenuntergang ist romantisch.',
    'Sterne funkeln am Himmel.',
    'Der Strand ist entspannend.',
    'Wellen rauschen sanft.',
    'Der Fluss fließt ruhig.',
    'Schmetterlinge sind farbenfroh.',
    'Der Garten blüht prächtig.',
    'Herbstblätter fallen.',
    'Schnee bedeckt die Landschaft.',
    'Frühling erweckt alles zum Leben.',
    'Sommer bringt Wärme.',
    'Winter ist kalt aber schön.',
    'Ich sammle Pilze im Wald.',
    
    # Gesundheit und Wohlbefinden
    'Ich fühle mich gut.',
    'Sport ist gesund.',
    'Ich ernähre mich ausgewogen.',
    'Schlaf ist wichtig.',
    'Ich meditiere täglich.',
    'Yoga entspannt mich.',
    'Ich trinke viel Wasser.',
    'Gesunde Ernährung ist wichtig.',
    'Ich gehe zum Arzt.',
    'Vorsorge ist wichtig.',
    'Ich nehme Vitamine.',
    'Bewegung hält fit.',
    'Ich atme tief durch.',
    'Entspannung ist notwendig.',
    'Ich höre auf meinen Körper.',
    'Gesundheit ist das Wichtigste.',
    'Ich mache Physiotherapie.',
    'Massagen sind wohltuend.',
    'Ich achte auf meine Haltung.',
    'Pausen sind wichtig.',
    
    # Hobbys und Freizeit
    'Ich sammle Briefmarken.',
    'Fotografie ist mein Hobby.',
    'Ich stricke gerne.',
    'Gartenarbeit entspannt mich.',
    'Ich spiele Schach.',
    'Musik ist meine Leidenschaft.',
    'Ich tanze gerne.',
    'Lesen bildet.',
    'Ich reise gerne.',
    'Kunst inspiriert mich.',
    'Ich koche leidenschaftlich.',
    'Theater ist kulturell bereichernd.',
    'Ich spiele ein Instrument.',
    'Handwerk macht Spaß.',
    'Ich restauriere alte Möbel.',
    'Nähen ist kreativ.',
    'Ich züchte Pflanzen.',
    'Angeln ist entspannend.',
    'Ich sammle Münzen.',
    'Puzzle lösen ist meditativ.'
]

# Alle Texte und Labels kombinieren
texts = tech_texts + non_tech_texts
labels = ['tech'] * len(tech_texts) + ['non-tech'] * len(non_tech_texts)

print(f"Anzahl Texte: {len(texts)}")
print(f"Anzahl Labels: {len(labels)}")
print(f"Tech-Beispiele: {labels.count('tech')}")
print(f"Non-Tech-Beispiele: {labels.count('non-tech')}")

# TF-IDF statt einfacher CountVectorizer für bessere Ergebnisse
# TF-IDF berücksichtigt die Wichtigkeit der Wörter
vectorizer = TfidfVectorizer(
    max_features=5000,  # Maximal 5000 wichtigste Wörter
    ngram_range=(1, 2),  # Einzelwörter und Wortpaare
    min_df=2,  # Wort muss mindestens 2x vorkommen
    stop_words=None  # Deutsche Stoppwörter würden zu viel filtern
)

x = vectorizer.fit_transform(texts)

# Daten aufteilen mit stratifizierter Aufteilung
x_train, x_test, y_train, y_test = train_test_split(
    x, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTrainingsdaten: {len(y_train)} Beispiele")
print(f"Testdaten: {len(y_test)} Beispiele")
print(f"Tech in Testdaten: {y_test.count('tech')}")
print(f"Non-Tech in Testdaten: {y_test.count('non-tech')}")

# Modell trainieren
model = MultinomialNB(alpha=0.1)  # Smoothing-Parameter angepasst
model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nGenauigkeit des Modells: {accuracy:.2%}")
print("\n--- Detaillierte Klassifikationsauswertung ---")
print(classification_report(y_test, y_pred))

# Funktion für bessere Vorhersagen mit Mindest-Sicherheit
def predict_with_confidence(text, min_confidence=0.6):
    """
    Vorhersage mit Mindest-Sicherheitsschwelle
    """
    x_new = vectorizer.transform([text])
    prediction = model.predict(x_new)[0]
    probabilities = model.predict_proba(x_new)[0]
    
    # Finde die Wahrscheinlichkeit für die vorhergesagte Klasse
    class_idx = list(model.classes_).index(prediction)
    confidence = probabilities[class_idx]
    
    tech_prob = probabilities[1] if model.classes_[1] == 'tech' else probabilities[0]
    non_tech_prob = probabilities[0] if model.classes_[0] == 'non-tech' else probabilities[1]
    
    # Wenn Sicherheit zu niedrig, als unsicher markieren
    if confidence < min_confidence:
        return 'unsicher', confidence, tech_prob, non_tech_prob
    else:
        return prediction, confidence, tech_prob, non_tech_prob

# Interaktive Vorhersage
print("\n" + "="*60)
print("--- Verbesserter Text-Klassifikator ---")
print("Gib Texte ein, die klassifiziert werden sollen.")
print("Beispiele: 'Ich programmiere Python', 'Ich esse Pizza'")
print("Oder 'quit' zum Beenden.")

while True:
    user_input = input("\nText eingeben: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Programm beendet.")
        break
    
    if not user_input:
        print("Bitte gib einen Text ein.")
        continue
    
    # Mehrere Texte unterstützen
    if ',' in user_input:
        texts_to_predict = [text.strip() for text in user_input.split(',') if text.strip()]
    else:
        texts_to_predict = [user_input]
    
    for text in texts_to_predict:
        prediction, confidence, tech_prob, non_tech_prob = predict_with_confidence(text)
        
        print(f"\nText: '{text}'")
        if prediction == 'unsicher':
            print(f"  → Kategorie: UNSICHER (zu niedrige Sicherheit: {confidence:.1%})")
        else:
            print(f"  → Kategorie: '{prediction}' (Sicherheit: {confidence:.1%})")
        print(f"  → Tech: {tech_prob:.1%}, Non-Tech: {non_tech_prob:.1%}")
        
        # Warnung bei sehr kurzen Texten
        if len(text.split()) <= 2:
            print("  ⚠️  Warnung: Sehr kurzer Text - Vorhersage kann unzuverlässig sein")