from django import forms


class BloodPressureForm(forms.Form):
    systolic = forms.IntegerField(min_value=50, max_value=260, label='Tension systolique (mmHg)')
    diastolic = forms.IntegerField(min_value=30, max_value=160, label='Tension diastolique (mmHg)')
