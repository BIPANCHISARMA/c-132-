import tensorflow.keras  
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ["I am happy to go on a holiday trip with my family.", 
            "I hate construction works in my neighbourhood."]

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentence)

# Create a word_index dictionary
word_index= tokenizer.word_index
sequence = tokenizer.texts_to_sequences(sentence)
print(sequence[0:2])

# Padding the sequence
padded = pad_sequences(sequence,maxlen = 100,padding='post', truncating='post')
print(padded[0:2])

# Define the model using .h5 file
model = tensorflow.keras.models.load_model('C:/Users/csarm/Downloads/-class-132-main/-class-132-main/Text_Emotion.h5')

# Test the model
result = model.predict(padded)
print(result)

# Print the result
predict_class= np.argmax(result, axis=1)
print(predict_class)

