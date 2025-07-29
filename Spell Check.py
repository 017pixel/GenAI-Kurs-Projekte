from textblob import TextBlob

text = str(input("Enter text to spell check: ")) # geht NUR englischer Text 

blob = TextBlob(text)

corrected_text = blob.correct()

print("Original Text:", text)
print("Corrected Text:", corrected_text)