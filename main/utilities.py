from django.template.loader import render_to_string
from django.core.signing import Signer
from datetime import datetime
from os.path import splitext
from scipy import spatial
from math import sqrt
from random import random
import numpy as np
from gensim.models import doc2vec
from collections import namedtuple

from bboard.settings import ALLOWED_HOSTS

SIZE = 200
NUM_CLUSTERS = 4
ITER = 20
signer = Signer()


def send_activation_notification(user):
    if ALLOWED_HOSTS:
        host = 'http://' + ALLOWED_HOSTS[0]
    else:
        host = 'http://localhost:8000'
    context = {'user': user, 'host': host, 'sign': signer.sign(user.username)}
    subject = render_to_string('email/activation_letter_subject.txt', context)
    body_text = render_to_string('email/activation_letter_body.txt', context)
    user.email_user(subject, body_text)


def get_timestamp_path(instance, filename):
    return '%s%s' % (datetime.now().timestamp(), splitext(filename)[1])


def AverageVector(vectors):                           # считает среднее векторов предлож.
    data = np.array(vectors)
    return np.average(data, axis=0)


def SentenceVectorDistance(vec1, vec2):               # расстояние между векторами
    return spatial.distance.cosine(vec1, vec2)


class AdVector:                                       # класс вектора объяв.
    def __init__(self, price, title, description):
        self.price = price
        self.title = title
        self.description = description

    def Distance(self, ad_vector):                                         # расстояние между объв.
        price_dist = 1.5 * (self.price - ad_vector.price)**2
        title_dist = SentenceVectorDistance(self.title, ad_vector.title)**2
        description_dist = SentenceVectorDistance(self.description, ad_vector.description)**2
        return sqrt(price_dist+title_dist+description_dist)

    def Average(self, ad_vectors):                                        # среднее векторов
        if len(ad_vectors) == 0:
            return self
        titles = []
        descriptions = []
        prices = 0
        for ad_vector in ad_vectors:
            prices += ad_vector.price
            titles.append(ad_vector.title)
            descriptions.append(ad_vector.description)
        length = len(ad_vectors)
        av_price = prices / length
        av_title = AverageVector(titles)
        av_description = AverageVector(descriptions)
        return AdVector(av_price, av_title, av_description)

    def Random(self, size):                                      # задаёт начальные точки класт.
        price = random()
        title = []
        description = []
        for i in range(0, size):
            title.append(random()*2 - 1)
            description.append(random()*2 - 1)
        return AdVector(price, title, description)


def GetCentroids(k, size):
    centroids = []
    adVector_empty = AdVector(0, [], [])
    for i in range(0, k):
        centroids.append(adVector_empty.Random(size))
    return centroids


def KMeans(ad_vectors, k, iterations):                        # кластеризация
    ad_vector_empty = AdVector(1, [], [])
    centroids = GetCentroids(k, SIZE)
    _claster_index = []
    _k_groups = []
    for i in range(0, iterations):
        claster_index = []
        for ad_vector in ad_vectors:
            min_dist = 100000000
            min_index = 0
            for i, centroid in enumerate(centroids):
                if centroid.Distance(ad_vector) < min_dist:
                    min_dist = centroid.Distance(ad_vector)
                    min_index = i
            claster_index.append(min_index)
        k_groups = []
        k_groups_ad_vectors = []
        for i in range(0, k):
            k_groups.append([])
            k_groups_ad_vectors.append([])
        for i, index in enumerate(claster_index):
            k_groups[index].append(i)
            k_groups_ad_vectors[index].append(ad_vectors[i])
        _k_groups = k_groups
        _claster_index = claster_index
        _centroids = []
        for i, group in enumerate(k_groups_ad_vectors):
            _centroids.append(centroids[i].Average(group))
        centroids = _centroids
    return _k_groups


def get_similar(bb, bbs):
    bbs_similar = []
    AdsTitle = []
    AdsDis = []
    AdsPrices = []
    kol = {}
    bb_pk = 0
    for i, bb1 in enumerate(bbs):
        AdsTitle.append(bb1.title)
        AdsDis.append(bb1.content)
        AdsPrices.append(bb1.price)
        kol[i] = bb1
        if bb1.pk == bb.pk:
            bb_pk = i
    doc1 = AdsTitle + AdsDis
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(doc1):
        words = text.lower().split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    model = doc2vec.Doc2Vec(docs, vector_size=SIZE, min_count=1, workers=4)
    ad_vectors = []
    for i, ad_title in enumerate(AdsTitle):
        ad_vectors.append(AdVector(AdsPrices[i] / max(AdsPrices),
                          model.docvecs[i], model.docvecs[i + len(AdsTitle)]))
    k = NUM_CLUSTERS
    k_groups = KMeans(ad_vectors, k, ITER)
    cluster = []
    for kluster in k_groups:
        if bb_pk in kluster:
            cluster = kluster
    for i in cluster:
        if i != bb_pk:
            bbs_similar.append(kol[i])
    bb_similar_sort_1 = []
    bb_similar_sort_2 = []
    bb_similar_sort_3 = []
    for bb1 in bbs_similar:
        if bb1.rubric == bb.rubric:
            bb_similar_sort_1.append(bb1)
        elif bb1.rubric.super_rubric == bb.rubric.super_rubric:
            bb_similar_sort_2.append(bb1)
        else:
            bb_similar_sort_3.append(bb1)
    bb_similar_sort = bb_similar_sort_1 + bb_similar_sort_2 + bb_similar_sort_3
    return bb_similar_sort
