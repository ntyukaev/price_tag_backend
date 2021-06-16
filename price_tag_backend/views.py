from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from .models import Timestamp


@csrf_exempt
def index(request):
    # 4859764
    id = '12345'
    with open('test.png', 'wb') as f:
        f.write(request.FILES['image'].read())
    vals = Timestamp.objects.filter(ID=id)
    prices = [v.price for v in vals]
    timestamps = [v.timestamp for v in vals]
    # data = [{'value': v.price, 'date': v.timestamp.strftime("%d-%b-%Y")} for v in vals]
    # print(data)
    print(prices)
    return JsonResponse({'data': prices})
