
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Embedding, GRU, Dense
from keras.src.optimizers import Adam
from keras.src.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

dataset=pd.read_csv("hepsiburada.csv")

target=dataset["Rating"].values.tolist()
data=dataset["Review"].values.tolist()

#Train ve testi ayırdık %80 olarak
cutoff=int(len(data)*0.8)
x_train=data[:cutoff]
x_test=data[cutoff:]

#Label çıktılarınıda dağıttık
y_train=target[:cutoff]
y_test=target[cutoff:]

""""
print(x_train[500])
print(y_train[500])

print(x_test[800])
print(y_test[800])
"""
#burada keras her bir yorumu gezecek ve en çok kulalnılan 10000 kelime İD verecek
num_words=10000
tokenizer=Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train)

"""
print(tokenizer.word_index)

for word, idx in list(tokenizer.word_index.items())[:20]:
    print(word, idx)
"""

#İD atılan rakamları artık string değil matematiksel duruma ceviriyoruz
x_train_tokens=tokenizer.texts_to_sequences(x_train)
x_test_tokens=tokenizer.texts_to_sequences(x_test)

#her bir oluşturulan yorum için token adet sayılarını satır satır sayılarını öğrenemk istiyoruz
num_tokens=[len(tokens) for tokens in x_train_tokens + x_test_tokens]

#numpy dizisine çevirdik çünkü bazı fonksiyonları kullanmamız daha kolay
num_tokens=np.array(num_tokens)
y_train = np.array(y_train)
y_test = np.array(y_test)
np.mean(num_tokens)

#tüm yorumları aynı uzunluğa getirmeliyiz =59 olucak
x_train_pad=pad_sequences(x_train_tokens,maxlen=59)

x_test_pad=pad_sequences(x_test_tokens,maxlen=59)

#modelimizi oluşturmaya başlıyoruz yapay sinir ağı oluşturduk
model = Sequential()

#bir id için gelecek context bağlam vektörleri
embedding_size=50

model.add(Embedding(input_dim=num_words,output_dim=embedding_size,input_length=59,name="embedding_layer"))

model.add(GRU(units=16,return_sequences=True))
model.add(GRU(units=8,return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1,activation='sigmoid'))

optimizer=Adam(learning_rate=1e-3) # learning_rate=0.001
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

model.fit(x_train_pad,y_train,epochs=3,batch_size=256)

model.save("lstm_model.keras")


model = load_model("lstm_model.keras")
predictions = model.predict(x=x_test_pad[0:1000])

predictions=predictions.T[0]


