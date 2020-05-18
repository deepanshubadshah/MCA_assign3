import nltk
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm


nltk.download('abc')

rwords = nltk.corpus.abc.words()[:15000]
words = []
for word in rwords:
  words.append(word.lower())

uwords = set(words)
vocab_size = len(set(words))
#print(len(words))
#print(vocab_size)

sentences = []
sentence = []
for word in words:
  if word == '.':
    sentences.append(sentence)
    sentence = []
  else:
    sentence.append(word)

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

data = []
WINDOW_SIZE = 5
word2int = {}
int2word = {}

for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])

for i,word in enumerate(uwords):
    word2int[word] = i
    int2word[i] = word

x_train = []
y_train = []
for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
#print(x_train.shape, y_train.shape)

EMBEDDING_DIM = 5
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))
hidden_representation = tf.nn.relu(tf.add(tf.matmul(x,W1), b1))

W1e = tf.Variable(tf.random_normal([EMBEDDING_DIM, EMBEDDING_DIM]))
b1e = tf.Variable(tf.random_normal([EMBEDDING_DIM]))
hidden_representation_e = tf.nn.relu(tf.add(tf.matmul(hidden_representation,W1e), b1e))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

# Training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 20
# train for n_iter iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss: ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

vectors = sess.run(W1 + b1)

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 100000000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

################## TSNE ############
keys = ['why', 'federation', 'past', 'farmers', 'country', 'the']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    similar_word = int2word[find_closest(word2int[word], vectors)]
    words.append(similar_word)
    #print(similar_word)
    embeddings.append(vectors[ word2int[similar_word] ])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)


embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Similar words', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')
