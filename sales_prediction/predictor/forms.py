from django import forms

class SalesDataForm(forms.Form):
    """Form for collecting sales data for prediction."""

    Store = forms.IntegerField(label="Store")
    DayOfWeek = forms.ChoiceField(
        label="Day of week",
        choices=[(1, "first day"), (2, "second day"), (3, "third day"),
                 (4, "fourth day"), (5, "fifth day"), (6, "sixth day"), (7, "seventh day")]
    )
    # sales = forms.IntegerField(label="Sales")
    Customers = forms.IntegerField(label="Customers")
    Date = forms.DateField(label="Date", widget=forms.TextInput(attrs={"placeholder": "YYYY-MM-DD"}))
    Open = forms.ChoiceField(
        label="Open status",
        choices=[(1, "Open"), (0, "Closed")]
    )
    Promo = forms.ChoiceField(
        label="Promo status",
        choices=[(1, "Promoted"), (0, "Not Promoted")]  # Corrected typo
    )
    StateHoliday = forms.ChoiceField(
        label="State holiday",
        choices=[('0', "No Holiday"), ('a', "a"), ('b', "b"), ('c', "c")]
    )
    SchoolHoliday = forms.ChoiceField(
        label="School holiday",
        choices=[(1, "School Holiday"), (0, "No School Holiday")]
    )