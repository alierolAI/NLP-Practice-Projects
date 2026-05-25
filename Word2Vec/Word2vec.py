import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

#word to word preprocessing
#haberleri ilk önce cümle cümle stringledik sonra liste oluşturup kelime kelime listeye koyduk
"""
f = open("hurriyet.txt", "r", encoding="utf-8")

text = f.read()
t_list=text.split('\n')

corpus = []

for t in t_list:
    corpus.append(t.split())
print(corpus[:10])
"""

#model training  modelimizi güzelce eğittik
"""
import multiprocessing

model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=5,
    sg=1,
    workers=max(1, multiprocessing.cpu_count() - 1)
)
model.save("word2vec.model")
"""

#kaydedilmiş eğitilmiş bir model üzerinde istediğimiz işlemleri yapabiiyoruz
"""
model = Word2Vec.load("word2vec.model")

print(model.wv.most_similar("hollanda"))
#ekran çıktısı
#[('avusturya', 0.8307461142539978), ('danimarka', 0.8140873312950134), ('belçika', 0.7794011831283569), ...vb]

print(model.wv.most_similar("türk"))
#ekran çıktısı: [('yunan', 0.6908259391784668), ('alman', 0.6704976558685303), ('rus', 0.6630839109420776), ...vb]
"""