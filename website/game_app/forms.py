from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

from .models import *
from django import forms


class RegisterUserForm(UserCreationForm):
    username = forms.CharField(max_length=20, label='Имя пользователя',
                               widget=forms.TextInput(attrs={'class': 'form-input'}))
    password1 = forms.CharField(label='Пароль', widget=forms.PasswordInput(attrs={'class': ''}))
    password2 = forms.CharField(label='Повторите пароль', widget=forms.PasswordInput(attrs={'class': ''}))

    class Meta:
        model = User
        fields = {'username', 'password1', 'password2'}
        widgets = {
            'username': forms.TextInput(attrs={'class': ''}),
            'password1': forms.PasswordInput(attrs={'class': ''}),
            'password2': forms.PasswordInput(attrs={'class': ''}),
        }


class LoginUserForm(AuthenticationForm):
    username = forms.CharField(label='Имя пользователя', widget=forms.TextInput(attrs={'class': ''}))
    password = forms.CharField(label='Пароль', widget=forms.PasswordInput(attrs={'class': ''}))


class ReviewForm(forms.ModelForm):
    class Meta:
        model = Reviews
        fields = ('text',)


class LibraryForm(forms.ModelForm):
    class Meta:
        model = Library
        fields = ('rate',)


class ProfileForm(forms.ModelForm):
    avatar = forms.ImageField(required=False, label='Иконка пользователя:')
    nickname = forms.CharField(min_length=4, max_length=30, required=False, label='Имя пользователя:')

    avatar.widget.attrs.update({'class': 'avatar'})
    nickname.widget.attrs.update({'class': 'nickname'})

    class Meta:
        model = Profile
        fields = ('avatar', 'nickname')


class TechSupportForm(forms.ModelForm):
    class Meta:
        model = TechSupport
        fields = ('question',)