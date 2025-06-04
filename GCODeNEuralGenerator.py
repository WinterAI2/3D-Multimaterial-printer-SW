# -*- coding: cp1251 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Flatten, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# Определим функцию для создания модели
def create_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 1000, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(1000, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# Обучающие данные
TextData = """ """
with open("H:\gcode for NN\GcodeTR.txt", "r") as txt:
     for i in txt.readlines():
         TextData += i


# Подготовка данных
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(TextData)
total_chars = len(tokenizer.word_index) + 1
max_sequence_len = 50

input_sequences = []
for i in range(0, len(TextData) - max_sequence_len, 1):
    sequence = TextData[i:i + max_sequence_len]
    input_sequences.append(sequence)

input_sequences = tokenizer.texts_to_sequences(input_sequences)
input_sequences = np.array(input_sequences)
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_chars)

# Создание и обучение модели
model = create_model(total_chars, max_sequence_len)
accuracy = 0
epochs = 0
while accuracy < 0.7:
    model.fit(xs, ys, epochs=1, verbose=1)
    loss, accuracy = model.evaluate(xs, ys, verbose=0)
    epochs += 1

# Сохранение модели
model.save('TextGenerator3000.h5')

# Генерация текста
def generate_text(seed_text, next_chars, model, max_sequence_len):
    generated_text = seed_text
    for _ in range(next_chars):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted_probs = model.predict(token_list)[0]
        predicted = np.argmax(predicted_probs)
        output_char = tokenizer.index_word.get(predicted, "")
        seed_text += output_char
        generated_text += output_char

    return generated_text

# Генерация текста с использованием модели
while True:
    seed_text = input("Вы: ")
    next_chars = 500
    generated_text = generate_text(seed_text, next_chars, model, max_sequence_len)
    print(generated_text)