from django.db import models


class Barcode(models.Model):
    ID = models.PositiveIntegerField(primary_key=True)
    UPCEAN = models.PositiveIntegerField()
    name = models.CharField(max_length=300)
    # category_id = models.PositiveIntegerField()
    # category_name = models.CharField(max_length=300)
    # brand_id = models.PositiveIntegerField()
    # brand_name = models.CharField(max_length=300)


class Timestamp(models.Model):
    identifier = models.AutoField(primary_key=True)
    ID = models.ForeignKey(Barcode, on_delete=models.CASCADE)
    price = models.PositiveIntegerField()
    timestamp = models.DateTimeField()
