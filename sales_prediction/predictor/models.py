from django.db import models

from django.db import models

class SalesData(models.Model):
    store = models.IntegerField()
    day_of_week = models.IntegerField()
    date = models.DateField()
    sales = models.FloatField()
    customers = models.IntegerField()
    open_status = models.IntegerField()
    promo = models.IntegerField()
    state_holiday = models.CharField(max_length=1)
    school_holiday = models.IntegerField()

    def __str__(self):
        return f"Store {self.store} - Date {self.date}"

