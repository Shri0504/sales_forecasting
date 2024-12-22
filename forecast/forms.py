from django import forms

class SalesForm(forms.Form):
    advertising = forms.FloatField(label='Advertising Budget', required=True)
    budget = forms.FloatField(label='Budget', required=True)
    competition = forms.FloatField(label='Competition', required=True)
