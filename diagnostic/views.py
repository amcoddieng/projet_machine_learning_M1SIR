from django.shortcuts import render
from .forms import BloodPressureForm
from .ml import predict_bp


def home(request):
    result = None
    if request.method == 'POST':
        form = BloodPressureForm(request.POST)
        if form.is_valid():
            sys = form.cleaned_data['systolic']
            dia = form.cleaned_data['diastolic']
            result = predict_bp(sys, dia)
    else:
        form = BloodPressureForm()

    return render(request, 'diagnostic/home.html', {
        'form': form,
        'result': result,
    })
