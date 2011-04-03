from django.conf.urls.defaults import *
from django.template import RequestContext
from django.http import HttpResponseRedirect

from indra.nodes.models import StoryNode, StoryEdge, NodeForm
from django.shortcuts import redirect, render_to_response
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required



# Create your views here.

# 

from django import forms
from django.contrib.auth.forms import UserCreationForm

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            new_user = form.save()
            return HttpResponseRedirect("/storynode/")
    else:
        form = UserCreationForm()
    return render_to_response("storynode/register.html", {
        'form': form,
        }, context_instance=RequestContext(request))
                                                    
def home_view(request):

    return render_to_response("storynode/storynode_base.html",
                              {'graph_type': 'summary'}, context_instance=RequestContext(request))

def filter_for_starts(nodes):
    result = []
    for n in nodes:
        ancestors = n.comes_from.all()
        if not ancestors:
            result.append(n)
    return result
    



def get_subgraph(node):
    sg = {}
    subgraph(node, sg)
    return sg

def subgraph(node, sg):
    children = node.goes_to.all()
    sg[node.id] = (node, children)
    if not children:
        # leaf node, done here
        return None
    else:
        # got some kids: add
        for e in children:
            if sg.has_key(e.to_node):
                continue
            else:
                subgraph(e.to_node, sg)
        return None


#def edit_node(request):
#    if request.method == 'POST':
#        form = NodeForm(request.POST)
#        if form.is_valid():
            



def author_page(request, username):
    author = User.objects.get(username__exact=username)
    full_object_list = author.storynode_set.all()
    object_list = filter_for_starts(full_object_list)
    total = 0
    # wordcounts = {}
    for n in object_list:
        node_wordcount = 0
        sg = get_subgraph(n)
        for k,v in sg.iteritems():

            spaces = v[0].body.split() + v[0].title.split()
            total += len(spaces)
            node_wordcount += len(spaces)
        # total += node_wordcount
        n.wordcount = node_wordcount
    return render_to_response(
        "storynode/storynode_list.html",
        {'object_list': object_list,
         'wordcount':total,
         'totalwordcount':total,
         'graph_type':'author', 'graph_target':username},
        # 'wordcounts':wordcounts},
        context_instance=RequestContext(request))



def get_summary_subgraph():
    sg = {}
    for n in StoryNode.objects.all():
        sg[n.id] = (n, n.goes_to.all())
    return sg

def get_author_subgraph(username):
    sg = {}
    author = User.objects.get(username__exact=username)
    for n in author.storynode_set.all():
        sg[n.id] = (n, n.goes_to.all())
    return sg



def node_graph_html(request, node_id):
    
    return render_to_response("storynode/nodegraph.html",{}, context_instance=RequestContext(request))

def author_graph_html(request, node_id):
    return render_to_response("storynode/authorgraph.html",{}, context_instance=RequestContext(request))
         




def build_graph(object_list):
    stuff = '[\n'
    for o,z in object_list.values():
        stuff += '{\n"adjacencies": [' + '\n   ' 
        stuff += ",\n   ".join( ['{"nodeTo":"%s", "nodeFrom":"%s", "data":{}}' %
                            (str(e.from_node.id), str(e.to_node.id) ) 
                            for e in o.comes_from.all()] )
        stuff += '\n],"data":{"$type":"circle","$dim":10},' + "\n"
        stuff += '"id":"%s", "name":"%s"},' % (str(o.id), str(o.title)) + "\n"
#    stuff = stuff[:-2]
    stuff = stuff[:-2] + "]" + "\n"    
    return stuff

def author_graph(request, username):

    
    return render_to_response("storynode/storygraph.html",
                              {'object_list': object_list,
                               'stuff':build_graph(object_list)})
# mimetype="application/javascript")

def graph(request, graph_type, graph_param):
    if graph_type == 'author':
        sg = get_author_subgraph(graph_param)
        #author = User.objects.get(username__exact=graph_param)
        #sg = author.storynode_set.all()        
        graph_title = "Author Graph for " + graph_param
    elif graph_type == 'summary':
        sg = get_summary_subgraph()
        graph_title = "Summary of all Nodes"
    elif graph_type == 'node':
        node = StoryNode.objects.get(pk=graph_param)
        sg = get_subgraph(node)
        graph_title = "StoryGraph for: " + node.title
    stuff = build_graph(sg)
    return render_to_response("storynode/storygraph.html",
                              {'graph_title': graph_title,
                               'stuff':stuff})
# mimetype="application/javascript")


    

    
@login_required
def attach_node(request, from_id):
    new_node = StoryNode(author=request.user)
    new_node.save()
    if from_id:
        from_node = StoryNode.objects.get(pk=from_id)
        new_edge = StoryEdge(from_node = from_node, to_node =  new_node)
        new_edge.save()
    return redirect(new_node.get_absolute_url()  + 'edit/')

@login_required
def handle_form(request, node_id):

    n = StoryNode.objects.get( pk=node_id )

    if n.author != request.user:
        return HttpResponseRedirect('/storynode/')
    if request.method == 'POST': # If the form has been submitted...
        form = NodeForm(request.POST) # A form bound to the POST data
        if form.is_valid(): # All validation rules pass
            # Process the data in form.cleaned_data
            n.title = form.cleaned_data['title']
            n.body = form.cleaned_data['body']
            n.reference_url = form.cleaned_data['reference_url']
            n.save()
            
            return HttpResponseRedirect(n.get_absolute_url()) # Redirect after POST
    else:
        form = NodeForm({'title':n.title, 'body':n.body}) # An unbound form
        #form.title = n.title
        #form.body = n.body
        
    return render_to_response('storynode/storynode_form.html', {
        'object': n, 'form':form}, context_instance=RequestContext(request)
        )
        
