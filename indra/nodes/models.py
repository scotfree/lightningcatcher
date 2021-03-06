from django.db import models
from django.contrib.auth.models import User
from django import forms

# Create your models here.


class Tag(models.Model):
    tag = models.CharField(max_length=200, blank=False)
    description = models.TextField(blank=True)
    
class StoryNode(models.Model):
    title = models.CharField(max_length=200, blank=True)
    body = models.TextField(blank=True)
    author = models.ForeignKey(User)
    modified_at = models.DateTimeField(auto_now=True)
    reference_url = models.TextField(blank=True)

    def __str__(self):
        return str( self.title )

    def get_absolute_url(self):
        return "/nodes/%i/" % self.id

class StoryEdge(models.Model):
    from_node = models.ForeignKey(StoryNode, related_name='goes_to')
    to_node = models.ForeignKey(StoryNode, related_name='comes_from')

    def __str__(self):
        return str( self.from_node ) + " --> " + str( self.to_node )

class NodeTag(models.Model):
    node = models.ForeignKey(StoryNode)
    tag = models.ForeignKey(Tag)

class NodeForm(forms.Form):
    title = forms.CharField(max_length=200, required=False,
                            widget=forms.TextInput(attrs={'size':'80'})
                            )
    reference_url = forms.CharField(max_length=300, required=False,
                            widget=forms.TextInput(attrs={'size':'80'})
                            )                            
    body = forms.CharField(widget=forms.Textarea(attrs={ 'cols':80}),
                           required=False
                           )
                           
    
