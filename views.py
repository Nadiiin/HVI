import args as args
from django.shortcuts import render
from django.http import HttpResponse
from  django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage


def home(request):
    return render(request,'blog/home.html')

def about(request):
    return render(request, 'about.html')
    #return HttpResponse('<h1>About</h1>')

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        #print(uploaded_file.name)
        #print(uploaded_file.size)
        fs = FileSystemStorage()
        fs.save(uploaded_file.name,uploaded_file)
    return render(request,'test/test.html')
def post(self,request):
    tmpl = 'test/test.html'
    return render(request,self.tmpl,args)