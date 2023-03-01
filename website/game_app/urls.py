from django.urls import path
from .views import *

urlpatterns = [
    path('', Index.as_view(), name='home'),
    path('profile/<slug:username>/', ProfileView.as_view(), name='profile'),
    path('signup', SignUpUser.as_view(), name='signup'),
    path('login', LoginUser.as_view(), name='login'),
    path('bd/', create),
    path('search/', search.as_view(), name='search'),
    path('logout/', LogOutUser, name="logout"),
    path('game/<slug:slug>/', GameView.as_view(), name='game'),
    path('<slug:slug>/', ReviewView.as_view(), name="post_review"),
    path('library/recommendation-list/', RecView.as_view(), name='rec'),
    path('support/send-question/', TechSupportView.as_view(), name='support'),
    path('support/subscribe/', SubscribeView.as_view(), name='subscribe'),
    path('chat/group/', ChatView.as_view(), name="chat"),

]
