from translate import Translator

translate = Translator(to_lang="es") # ist Spanish
text = "I am learning!" # Text geht nur von Englisch zu andere Sprache

translation = translate.translate(text)

print("")
print("Original Text:", text)
print("Translated Text:", translation)
print("")