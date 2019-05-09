import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
import json
import random
from nltk.corpus import stopwords

app = Flask(__name__)

f = open("corpus.txt", 'r')
data = f.read()
sent_tokens = nltk.sent_tokenize(data)
greeting_keywords = ['hi', 'hello', 'yo', 'howdy', 'hiya']
greeting_response = ['hi', 'hello', 'yo', 'howdy', 'hiya']


def remove_stopwords(sentence):
    stop_words = set(stopwords.words("english"))
    word_tokens = nltk.word_tokenize(sentence)

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    filter_string = ""
    count = 0

    for w in filtered_sentence:
        count += 1
        if count == 1:
            filter_string += str(w)
        else:
            filter_string += " " + str(w)

    return filter_string


def check_greeting(inp):
    if inp in greeting_response:
        return str(random.choice(greeting_response))


def gen_resp(inp):
    query_final = remove_stopwords(inp)
    sent_tokens.append(query_final)

    if check_greeting(query_final) is None:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        if req_tfidf == 0:
            return "invalid input"
        else:
            resp = sent_tokens[idx].split(": ")
            return str(resp[1])
    else:
        return check_greeting(inp)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/process_data", methods=['GET'])
def process_data():
    message = request.args.get('inp1')
    mydict = {"message": gen_resp(message)}
    sent_tokens.pop()
    return json.dumps(mydict)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
