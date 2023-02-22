# Create your views here.

# Create your views here.
import time

from django.core.files.storage import FileSystemStorage
from django.shortcuts import HttpResponse, render
from django.views.decorators.csrf import csrf_exempt

from app.abnormal_activity.core import train_from_video, test_from_video
from app.settings import MEDIA_ROOT


# Create your views here.


def index(request):
    results = {
        "success": True
    }
    return render(request, 'index.html',results)


def save_file(input_img):
    fs = FileSystemStorage()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = str(timestr) + "_" + input_img.name
    file_name = file_name.replace(' ', '_')
    filename = fs.save(file_name, input_img)
    uploaded_file_url = fs.url(filename)
    local_file_dir = MEDIA_ROOT + file_name
    print('uploaded_file_url', uploaded_file_url, local_file_dir)
    return uploaded_file_url, local_file_dir, filename


def train_classifier(request):
    results = {
        "success": False
    }
    print = "train_classifier"
    if request.method == 'GET':
        trainingSet = ['train1']
        for video_name in trainingSet:
            video_path = MEDIA_ROOT + 'input\\train\\{}.avi'.format(video_name)
            results['success'] = train_from_video(video_path, video_name)


        vid = MEDIA_ROOT + 'input\\train\\test1.avi'
        test_from_video(vid, 'test1.avi')
        print("Done")

    return HttpResponse('index.html', results)

@csrf_exempt
def detect_abnormal_activity(request):
    results = {
        "success": False,
        "unusualFramesCount": 0,
        "listOfUnusualFrames": []
    }
    print('detect_abnormal_activity', request.FILES)
    if request.method == 'POST' and 'video_file' in request.FILES:
        uploaded_file_url, local_file_dir, filename = save_file(request.FILES['video_file'])
        filename = filename.replace('.avi','')
        success, unusualFramesCount, listOfUnusualFrames = test_from_video(local_file_dir, filename)
        results = {
            "success": success,
            "unusualFramesCount": unusualFramesCount,
            "file": listOfUnusualFrames[0]
        }
        print(results)

    return HttpResponse('index.html', results)
