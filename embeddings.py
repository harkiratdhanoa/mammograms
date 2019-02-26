from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

import nltk

from nltk.corpus import stopwords

stopw = stopwords.words('english')
stopw.append('also')

def readFile(file):
    f=open(file,'r',encoding='utf-8')
    text=f.read()
    
    sentences=nltk.sent_tokenize(text)
    
    data = []
    
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        words = [w.lower() for w in words if len(w)>2 and w not in stopw]
        data.append(words)
        
    return data

sentences = readFile("bollywood.txt")

# train model
model = Word2Vec(sentences, min_count=1)
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['actress'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)


# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

