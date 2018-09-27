from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from gensim import corpora, models, similarities
from pprint import pprint
from bs4 import BeautifulSoup
import warnings
import linecache
import math
import urllib3


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
doc_f = open('javadoc_dict.txt', 'r')
doc_str = doc_f.read()
javadoc_dict = eval(doc_str)
doc_f.close()
http = urllib3.PoolManager()


def core_search(cat, query, page=1):
    t_dictionary = corpora.Dictionary.load(cat+"t_dict.dict")
    p_dictionary = corpora.Dictionary.load(cat+"p_dict.dict")
    c_dictionary = corpora.Dictionary.load(cat+"c_dict.dict")

    tfidf_model = cat + 'tfidf_for_LDA.model'
    code_trained_model = cat + 'code_trained_LDA_model.model'
    code_tfidf_model = cat + 'code_tfidf_for_LDA.model'
    title_tfidf_model = cat + 'title_tfidf_for_LDA.model'
    code_trained_index = cat + 'code_trained_LDA_index.index'

    stopwords_file = open(cat + 'unigram_stops', 'r', encoding='utf-8')

    dictionary = p_dictionary
    code_dictionary = c_dictionary
    title_dictionary = t_dictionary

    tfidf = models.TfidfModel.load(tfidf_model)
    code_lda = models.LdaModel.load(code_trained_model)
    code_tfidf = models.TfidfModel.load(code_tfidf_model)
    title_tfidf = models.TfidfModel.load(title_tfidf_model)

    ret = []
    q_tokenized = [word.lower() for word in word_tokenize(query)]
    english_stopwords = [word.strip('\n') for word in stopwords_file]
    stopwords_file.close()
    q_stemmed = []
    st = PorterStemmer()
    for word in q_tokenized:
        is_soy = True
        for i in range(len(word)):
            if word[i].isalpha():
                is_soy = False
        if not is_soy:
            q_stemmed.append(st.stem(word))
    q_filterer_stop = [word for word in q_stemmed if word not in english_stopwords]
    print(q_filterer_stop)
    q_bow = dictionary.doc2bow(q_filterer_stop)
    code_q_bow = code_dictionary.doc2bow(q_filterer_stop)
    title_q_bow = title_dictionary.doc2bow(q_filterer_stop)

    q_tfidf = tfidf[q_bow]
    code_q_lda = code_lda[code_q_bow]
    code_q_tfidf = code_tfidf[code_q_bow]
    title_q_tfidf = title_tfidf[title_q_bow]

    tfidf_index = similarities.Similarity.load(cat + 'trained_tfidf_for_LDA_index.index')
    tfidf_sims = tfidf_index[q_tfidf]
    print(type(tfidf_sims))
    del tfidf_index
    code_tfidf_index = similarities.Similarity.load(cat + 'code_trained_tfidf_for_LDA_index.index')
    code_tfidf_sims = code_tfidf_index[code_q_tfidf]
    del code_tfidf_index
    title_tfidf_index = similarities.Similarity.load(cat + 'title_trained_tfidf_for_LDA_index.index')
    title_tfidf_sims = title_tfidf_index[title_q_tfidf]
    del title_tfidf_index
    code_index = similarities.Similarity.load(code_trained_index)
    code_lda_sims = code_index[code_q_lda]
    del code_index

    filtered_num = [sim[0] for sim in enumerate(code_lda_sims) if sim[1] <= 0.7]
    sims = []

    for num in filtered_num:
        tfidf_sims[num] = 0.0
        code_tfidf_sims[num] = 0.0
        title_tfidf_sims[num] = 0.0

    for i in range(len(tfidf_sims)):
        a = code_tfidf_sims[i]
        b = tfidf_sims[i]
        c = title_tfidf_sims[i]
        if a < 0.5 and b > 0.8:
            b = 0.8
        sims.append((i, 2*a+b+4*c))
    sorted_sims = sorted(sims, key=lambda item: -item[1])

    i = 10*(page-1)
    while i < 10*page:
        line_num = sorted_sims[i][0] + 1
        i = i + 1
        code_line = linecache.getline(cat + 'code', line_num)
        if(len(code_line) < 20):
            continue
        post_line = linecache.getline(cat + 'body', line_num)
        title_line = linecache.getline(cat + 'title', line_num)
        id_line = linecache.getline(cat + 'id', line_num)
        temp_dic = {}
        temp_dic['id'] = id_line.strip('\n')
        temp_dic['title'] = title_line.strip('\n')
        temp_dic['post'] = post_line.strip('\n')
        temp_dic['code'] = code_line.strip('\n')
        ret.append(temp_dic)
    return ret


def Word_synonyms(word):
    synonyms = []
    list_good = wordnet.synsets(word)
    for syn in list_good:
        for l in syn.lemmas():
            synonyms.append(l.name())
    return set(synonyms)


def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def associate(cat, keyword):
    lda = models.LdaModel.load(cat + "trained_LDA_model.model")
    dictionary = corpora.Dictionary.load(cat + "p_dict.dict")
    aret = []
    for word in Word_synonyms(keyword):
        aret.append(word)
    print("wordnet:")
    pprint(Word_synonyms(keyword))
    t2id = dictionary.token2id
    pprint(dictionary.get(113))
    if keyword not in t2id.keys():
        return aret
    kw_id = t2id[keyword]
    topic_num = len(lda.print_topics(-1, 1))
    temp_topics = lda.get_term_topics(kw_id)
    topic_vec = []
    for i in range(topic_num):
        topic_vec.append(0)
    for tp in temp_topics:
        topic_vec[tp[0]] = tp[1]
    print("vec0")
    pprint(topic_vec)
    sims = []
    for id in t2id.values():
        if id == kw_id:
            continue
        t_topic = lda.get_term_topics(id)
        if len(t_topic) == 0:
            continue
        t_vec = []
        for i in range(topic_num):
            t_vec.append(0)
        for tp in t_topic:
            t_vec[tp[0]] = tp[1]
        if id == 613:
            print("vec1")
            pprint(t_vec)
        t_sim = cosine_similarity(topic_vec, t_vec)
        sims.append((id, t_sim))

    sorted_sims = sorted(sims, key=lambda item: -item[1])
    print("top 30 sims")
    i = 0
    while i < 30:
        pprint(sorted_sims[i])
        i += 1
    for sim_record in sorted_sims:
        if sim_record[1] >= 0.9:
            try:
                print(sim_record[0])
                word = dictionary.id2token[sim_record[0]]
                print(word+":")
                pprint(sim_record)
                aret.append(word)
                print("add success")
            except Exception as e:
                pprint(e)
                continue
        else:
            break
    pprint(aret)
    return aret


def get_doc(name):
    ret = {}
    if name in javadoc_dict:
        url = 'https://docs.oracle.com/javase/8/docs/api/' + javadoc_dict[name]
        r = http.request('GET', url)
        soup = BeautifulSoup(r.data, 'lxml')
        description = soup.find('div', class_='block').get_text()
        ret['found'] = True
        ret['url'] = url
        ret['description'] = description
    else:
        ret['found'] = False
    return ret
