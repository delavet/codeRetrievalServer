from django.http import HttpResponse
import simplejson
from . import engine


def hello(request):
    return HttpResponse("Hello world ! ")


def inner_search(cat, datas, page):
    ret = engine.core_search(cat, datas, page)
    return ret


def search(request):
    dic = {}
    info = "success"
    try:
        if request.method == 'POST':
            req = simplejson.loads(request.body)
            cat = req["cat"]
            datas = req["query"]
            pg = req["page"]
            dic['result'] = inner_search(cat, datas, pg)
    except Exception:
        import sys
        info = "%s || %s" % (sys.exc_info()[0], sys.exc_info()[1])

    dic["message"] = info
    json = simplejson.dumps(dic)
    return HttpResponse(json)


def asso(request):
    dic = {}
    info = "success"
    try:
        if request.method == 'POST':
            req = simplejson.loads(request.body)
            cat = req["cat"]
            kw = req["keyword"]
            dic['result'] = engine.associate(cat, kw)
    except Exception:
        import sys
        info = "%s || %s" % (sys.exc_info()[0], sys.exc_info()[1])
    dic['message'] = info
    json = simplejson.dumps(dic)
    return HttpResponse(json)


def find_desc_in_doc(request):
    dic = {}
    try:
        if request.method == 'POST':
            req = simplejson.loads(request.body)
            name = req["name"]
            dic = engine.get_doc(name)
    except Exception:
        dic['found'] = False
    json = simplejson.dumps(dic)
    return HttpResponse(json)
