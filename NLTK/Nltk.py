#first code nltk Tokenization
"""
 eğer bunu yazmassan tokenize işlemini yapamıyor
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize,word_tokenize

text="Alan Mathison Turing (23 Haziran 1912 – 7 Haziran 1954), İngiliz matematikçi, bilgisayar bilimcisi ve kriptolog. Bilgisayar biliminin kurucusu sayılır. Geliştirmiş olduğu Turing testi ile makinelerin ve bilgisayarların düşünme yetisine sahip olup olamayacakları konusunda bir kriter öne sürmüştür."

#virgülleride alıyorken alttaki word_tokenize almıyor daha doğru yapıyor
print(text.split())
print(word_tokenize(text))

#If you want, you can tokenize the text into sentences.
print(sent_tokenize(text))
#The text was tokenized into three sentences.

#if you want to see it by line by
for token in word_tokenize(text):
    print(token)
"""


#Second code Nltk Stopwords
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#if you want to see stopwords iy by line by
for token in stopwords.words('english'):
    print(token)

#if you want to see turkish stopword
for string in stopwords.words('turkish'):
    print(string)

text="Türkiye İş Bankası AŞ, kısaca İş Bankası, Türkiye'de bireysel ve ticari bankacılık hizmeti sunan en büyük özel bankadır.[3] 1924'te Mustafa Kemal Atatürk'ün emriyle kurulan İş Bankası, cumhuriyet döneminin ilk ulusal bankasıdır.[4] 2023'te 2,5 trilyon Türk lirası aktif büyüklüğe[3] ulaşan banka, The Banker dergisinin açıkladığı sıralamaya göre ana sermaye bazında dünyanın en büyük 181. bankasıdır. Aralık 2021 itibarıyla 21'i yurt dışında olmak üzere 1.195 şube, 6.476 bankamatik ve 22 bini aşkın personeli bulunan İş Bankası'nın ayrıca Almanya'da İşbank AG, Gürcistan'da İşbank Georgia AŞ ve Rusya'da İşbank AŞ adıyla üç iştiraki vardır."

print(text)
stopwords=stopwords.words('turkish')
words=word_tokenize(text)

filtered_words=[]
for word in words:
    if word not in stopwords:
        filtered_words.append(word)
print(filtered_words)
"""

##3) Stemming
"""
from nltk.stem import PorterStemmer

ps = PorterStemmer()

words=["drive","driving","child","children","cats","dog"]

for word in words:
    print(ps.stem(word))
"""

#Part of speech tagging
"""
import nltk
nltk.download('averaged_perceptron_tagger_eng')
text="Ali plays football"

token=nltk.word_tokenize(text)
print(nltk.pos_tag(token))
"""

#NER -Named Entity Recognition
"""
import nltk
text="At 6:37 a.m. EDT, the SpaceX Dragon spacecraft docked to the forward port of the International Space Station’s Harmony module, carrying nearly 6,500 pounds of food, supplies, and equipment for the Expedition 74 crew. This is the 34th SpaceX commercial resupply services mission to the space station for NASA."
token=nltk.word_tokenize(text)
tagged=nltk.pos_tag(token)
chunk=nltk.ne_chunk(tagged)
chunk.draw()
"""

"""
#stemming kelimeye eklerine odaklanırken lemma dil yapısına göre yakalayabiliyor
#lemma daha iyi stemming den
#lemmatizing
from nltk.stem import WordNetLemmatizer
lemm=WordNetLemmatizer()

words=["drive","driving","drove","child","children","cats","dog","man"]

for word in words:
    print(lemm.lemmatize(word))
"""


