import asyncio
from datetime import datetime
from typing import Any
from django.contrib.auth import logout, login
from django.contrib.auth.views import LoginView
from django.db.models import Q
from django.db.models.functions import Lower
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views import View
from django.views.decorators.http import require_http_methods
from django.views.generic import ListView, CreateView, DetailView
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from sklearn.cluster import KMeans
import time
from game_app.utils import *
from game_app.models import *
from game_app.forms import *
import multiprocessing
from fill_db import get_similar
from rec_func import analyze_comment
import math
import pandas as pd
from django_pandas.io import read_frame

from django.core.mail import EmailMultiAlternatives, send_mail
from django.template.loader import render_to_string
from django.conf import settings


def LogOutUser(request):
    logout(request)
    return redirect('home')


class Index(ListView):
    model = Game
    template_name = 'game_app/index.html'

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        context['popular'] = Game.objects.filter(
            rating_count__gt=0).order_by('-rating_count', '-release_date')[:6]
        context['recently'] = Game.objects.filter(release_date__lte=datetime.now()).order_by('-release_date')[:6]
        context['waiting'] = Game.objects.filter(release_date__gt=datetime.now()).order_by('-release_date').order_by(
            'release_date')[:5]
        return context

    def get_success_url(self):
        return reverse_lazy('home')


class GameView(DetailView):
    model = Game
    template_name = 'game_app/game.html'
    slug_field = 'slug'
    slug_url_kwarg = 'slug'

    @staticmethod
    def get_similar_games():
        data = pd.DataFrame(Game.objects.values())
        data = data[:300]
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(data)
        data['label'] = kmeans.labels_
        return data["label"].query("label == 1").head(10)

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        context["similar"] = self.get_similar(self.object.game_id)
        context['back_img'] = Images.objects.filter(game_id=self.object.game_id)[1]
        context['websites'] = Websites.objects.filter(game_id=self.object.game_id)
        context['images'] = Images.objects.filter(game_id=self.object.game_id)[1:]
        context['videos'] = Videos.objects.filter(game_id=self.object.game_id)
        try:
            context['prev'] = Library.objects.get(user=self.request.user, game=self.object)
        except:
            pass
        context["reviews"] = Reviews.objects.filter(game=self.object).order_by("-date")[:5]
        return context

    @staticmethod
    def get_similar(game_id):
        games = get_similar(game_id)
        return Game.objects.filter(game_id__in=games)


class ProfileView(DetailView):
    model = User
    template_name = 'game_app/profile.html'
    slug_field = 'username'
    slug_url_kwarg = 'username'

    def post(self, request, username):
        profile_bd = Profile.objects.get(user=request.user)
        profile = ProfileForm(self.request.POST, request.FILES, instance=request.user.profile,
                              initial={'nickname': profile_bd.nickname, 'avatar': profile_bd.avatar})
        if profile.is_valid():
            profile.save()
        return redirect('profile', username)

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)

        context['profile'] = Profile.objects.get(user=context['object'])
        context['library'] = Library.objects.filter(user=context['object'])
        context['game_count'] = len(list(Library.objects.filter(user=context['object'])))
        context['review_count'] = len(list(Library.objects.filter(user=context['object'], review__isnull=False)))
        context['rate_count'] = len(list(Library.objects.filter(user=context['object'], rate__isnull=False)))
        context['rates'] = [len(list(Library.objects.filter(user=context['object'], rate=i))) for i in range(1, 11)]
        pc = 0
        ps = 0
        android = 0
        ios = 0
        linux = 0
        sega = 0
        xbox = 0
        nintendo = 0
        total = 0
        for game in Library.objects.filter(user=context['object']):
            for plat in game.game.platforms.all():
                if 'PC' in plat.name:
                    pc += 1
                if 'Linux' in plat.name:
                    linux += 1
                if 'Play' in plat.name:
                    ps += 1
                if 'Android' in plat.name:
                    android += 1
                if 'iOS' in plat.name:
                    ios += 1
                if 'Nintendo' in plat.name:
                    nintendo += 1
                if 'Sega' in plat.name:
                    sega += 1
                if 'Xbox' in plat.name:
                    xbox += 1
                total += 1
        if total > 0:
            context['pc_count'] = pc / total
            context['ps_count'] = ps / total
            context['ios_count'] = ios / total
            context['android_count'] = android / total
            context['linux_count'] = linux / total
            context['sega_count'] = sega / total
            context['nintendo_count'] = nintendo / total
            context['xbox_count'] = xbox / total
        # print(context['nintendo_count'], context['xbox_count'])
        context['settings_form'] = ProfileForm
        return context


class LoginUser(DataMixin, LoginView):
    form_class = AuthenticationForm
    template_name = 'game_app/login.html'
    success_url = reverse_lazy('home')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context()
        return dict(list(context.items()) + list(c_def.items()))

    def get_success_url(self):
        return reverse_lazy('home')


class SignUpUser(DataMixin, CreateView):
    form_class = UserCreationForm
    template_name = 'game_app/register.html'
    success_url = reverse_lazy('home')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context()
        return dict(list(context.items()) + list(c_def.items()))

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('home')


class search(ListView):
    # model = Game
    template_name = 'game_app/search.html'
    paginate_by = 50

    def get_queryset(self):
        rate = self.request.GET.get('rate')
        alp_asc = self.request.GET.get('alp_asc')
        name = self.request.GET.get('name')
        start = self.request.GET.get('start')
        end = self.request.GET.get('end')
        if end and start and end < start:
            start, end = end, start
        genres = self.request.GET.get('genres').split(',')[:-1] if self.request.GET.get('genres') else []
        genres = Genres.objects.filter(name__in=genres)
        platforms = self.request.GET.get('platforms').split(',')[:-1] if self.request.GET.get('platforms') else []
        platforms = Platforms.objects.filter(name__in=platforms)
        developers = self.request.GET.get('developers').split(',')[:-1] if self.request.GET.get('developers') else []
        game = Game.objects.none()
        if name is not None:
            if alp_asc == 'False':
                if rate:
                    game = Game.objects.filter(name__icontains=name).order_by(Lower('name').desc()).order_by('rating')
                else:
                    game = Game.objects.filter(name__icontains=name).order_by(Lower('name').desc()).order_by('-rating')
            else:
                if rate:
                    game = Game.objects.filter(name__icontains=name).order_by(Lower('name').asc()).order_by('rating')
                else:
                    game = Game.objects.filter(name__icontains=name).order_by(Lower('name').asc()).order_by('-rating')

            if start:
                game = game.exclude(Q(release_date__lt=start) | Q(release_date__isnull=True))
            if end:
                game = game.exclude(Q(release_date__gt=end) | Q(release_date__isnull=True))
            if list(genres):
                for i in list(genres):
                    game = game.exclude(~Q(genres=i))
            if list(platforms):
                for i in list(platforms):
                    game = game.exclude(~Q(platforms=i))
            if developers:
                for i in developers:
                    game = game.exclude(~Q(developer=i))

        return game

    def get_context_data(self, *, object_list=None, **kwargs):
        text = self.request.GET.get('name')
        context = super().get_context_data(**kwargs)
        page = self.request.GET.get('page', '1')
        context['page_range'] = context['paginator'].get_elided_page_range(number=page, on_ends=1, on_each_side=1)
        context['get_req'] = self.request.GET
        context['genres'] = Genres.objects.all()
        context['platforms'] = Platforms.objects.all()
        context['developers'] = Game.objects.values('developer').distinct().order_by('developer')
        return context


def create(request):
    offset = 107655
    while offset < 230_000:
        i = 0
        games = get_game(offset)
        try:
            for game in games:
                print(game["game_id"])
                i += 1  # 1
                # for j in game:
                #     print(j, ' : ', game[j])
                i += 1  # 2
                new_game = Game(game_id=game["game_id"], name=game["name"], slug=game["slug"],
                                developer=game["developer"],
                                description=game["description"], rating=int(game["rating"]),
                                rating_count=int(game["rating_count"]))
                if game["cover"]:
                    new_game.cover = game["cover"]
                if game["release_date"]:
                    new_game.release_date = game["release_date"]
                new_game.save()
                i += 1  # 3
                new_game.genres.set(Genres.objects.filter(name__in=game["genres"]))
                i += 1  # 4
                new_game.platforms.set(Platforms.objects.filter(name__in=game["platforms"]))
                i += 1  # 5
                new_game.save()
                i += 1  # 6
                if len(game["websites"]) > 0:
                    for el in game["websites"]:
                        if len(list(Websites.objects.filter(game_id=game["game_id"], name=el,
                                                            url=game["websites"][el]))) == 0:
                            Websites(game_id=game["game_id"], name=el, url=game["websites"][el]).save()
                i += 1  # 7
                if len(game["release_dates"]) > 0:
                    for el in game["release_dates"]:
                        if len(list(ReleaseDates.objects.filter(game_id=game["game_id"], platform=el))) == 0:
                            rel_date = ReleaseDates(game_id=game["game_id"], platform=el)
                            if game["release_dates"][el]:
                                rel_date.date = game["release_dates"][el]
                            rel_date.save()
                i += 1  # 8
                if len(game["video"]) > 0:
                    for el in game["video"]:
                        if len(list(Videos.objects.filter(game_id=game["game_id"], video_id=el))) == 0:
                            Videos(game_id=game["game_id"], video_id=el).save()
                i += 1  # 9
                if len(game["images"]) > 0:
                    for el in game["images"]:
                        if len(list(Images.objects.filter(game_id=game["game_id"], image_id=el))) == 0:
                            Images(game_id=game["game_id"], image_id=el).save()
                i = 0
                print('end')
        except:
            print('error', i)
            i = 0
            break
        else:
            offset += 500

    context = {}
    return render(request, 'game_app/bd.html', context=context)


class ReviewView(View):
    def post(self, request, slug):
        game = Game.objects.get(slug=slug)
        text = request.POST.get('text')
        rate = request.POST.get('rate', None)

        try:
            review = Reviews.objects.get(game=game, user=request.user)
            # print('get suc')
            review.text = text
            # print('update suc')
            review.save()
        except:
            review = Reviews()
            review.game = game
            review.text = text
            review.user = request.user
            review.save()

        try:
            library = Library.objects.get(user=request.user, game=game)
            library.review = review
            if rate:
                library.rate = rate
            library.save()
        except:
            library = Library()
            library.game = game
            library.user = request.user
            library.review = review
            if rate:
                library.rate = rate
            library.save()

        return redirect("game", slug)


class RecView(ListView):
    template_name = 'game_app/recommendation.html'

    @staticmethod
    def get_format_dict(arr):
        mentions = dict()
        for el in arr:
            if el.user not in mentions:
                mentions[el.user] = dict()
            if el.rate is not None:
                mentions[el.user][el.game] = el.rate
            else:
                mentions[el.user][el.game] = 5
        return mentions

    @staticmethod
    def distCosine(vecA, vecB):
        def dotProduct(vecA, vecB):
            d = 0.0
            for dim in vecA:
                if dim in vecB:
                    d += vecA[dim] * vecB[dim]
            return d

        return dotProduct(vecA, vecB) / math.sqrt(dotProduct(vecA, vecA)) \
               / math.sqrt(dotProduct(vecB, vecB))

    @classmethod
    def makeMatches(self, userID, userRates, nBestUsers, nBestProducts):
        matches = [(u, self.distCosine(userRates[userID], userRates[u])) for u in userRates if u != userID]

        bestMatches = sorted(matches, key=lambda x: x[1], reverse=True)[:nBestUsers]

        sim = dict()
        sim_all = sum([x[1] for x in bestMatches])
        bestMatches = dict([x for x in bestMatches if x[1] > 0.0])
        # print(bestMatches) #users
        for relatedUser in bestMatches:
            for product in userRates[relatedUser]:
                if product not in userRates[userID]:
                    if product not in sim:
                        sim[product] = 0.0
                    sim[product] += userRates[relatedUser][product] * bestMatches[relatedUser]
        for product in sim:
            sim[product] /= sim_all
        bestProducts = sorted(sim.items(), key=lambda x: x[1], reverse=True)[:nBestProducts]  # games
        # return [(x[0], x[1]) for x in bestProducts]
        return {'games': [x[0] for x in bestProducts], 'users': [x for x in bestMatches]}

    @classmethod
    def makeRecommendations(self, matches):
        rec_list = {'users': [], "games": [], 'coefs': []}
        urls = []
        games = []
        no_url_games = []
        for game in matches['games']:
            try:
                urls.append(Websites.objects.get(game_id=game.game_id, name='steam').url)
                games.append(game)
            except:
                no_url_games.append(game.game_id)

        probas = analyze_comment(urls)
        for index in range(len(probas)):
            rec_list["games"].append(games[index].game_id)
            rec_list["coefs"].append(probas[index])

        rec_list['games'] += no_url_games
        for user in matches["users"]:
            rec_list["users"].append(user.username)
        return rec_list

    def get_queryset(self):
        return Library.objects.all()

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        recs = self.makeRecommendations(
            self.makeMatches(self.request.user, self.get_format_dict(self.object_list), 20, 20))
        context['games'] = Game.objects.filter(game_id__in=recs["games"])
        context['users'] = User.objects.filter(username__in=recs["users"])
        text = '''<p><span>РЕКОМЕНДОВАНЫЕ ИГРЫ</span>, СОСТАВЛЕННЫЕ ПО ВАШЕЙ ЛИЧНОЙ БИБЛИОТЕКЕ ИГР.</p>
                    <p>В РАСЧЁТЕ УЧАВСТВОВАЛИ <span>ПОЛЬЗОВАТЕЛИ</span>, С КОТОРЫМИ У ВАС СХОЖИ ИНТЕРЕСЫ.</p>
                    <p>ПРИ ЭТОМ УЧИТЫВАЛИСЬ <span>ОТЗЫВЫ</span> НА КАЖДУЮ ИГРУ.</p>
                    <p>НАДЕЕМСЯ, ЧТО ВЫ ДОВОЛЬНЫ РЕКОМЕНДАЦИЯМИ</p>'''
        for index in range(len(context['games'])):
            if index < len(recs['coefs']):
                text += f'<b>{context["games"][index].name}</b>:\tанализ отзывов на эту дал оценку - \t<b>{recs["coefs"][index] * 100 // 1}%</b> негативных сообщений;<br><br> '
            else:
                text += f'<b>{context["games"][index].name}</b>:\tанализ отзывов не проводился(Отзывы недоступны) <br><br>'
        send(self.request.user.email, text)
        return context


def send(email, text):
    data = {
        'topic': f'Ваш список рекомендаций',
        'text': text,
        'user': 'Сайт',
    }
    html_body = render_to_string('game_app/email_template.html', data)
    msg = EmailMultiAlternatives(data['topic'], html_body, from_email=settings.EMAIL_HOST_USER,
                                 to=[email, ])
    msg.content_subtype = "html"
    msg.send()
    # print('email done')


class TechSupportView(View):
    def get(self, request):
        topics = MessageTopics.objects.all()
        selected = []
        for topic in topics:
            if request.user in topic.users.all():
                selected.append(topic)
        context = {'topics': topics, 'selected': selected}
        return render(request, template_name='game_app/TechSupport.html', context=context)

    def post(self, request):
        if request.POST.get('topic') and request.POST.get('text'):
            data = {
                'topic': request.POST.get('topic'),
                'text': request.POST.get('text'),
                'user': request.user.email,
            }
            message = TechSupport()
            message.question = data['text']
            message.user = request.user
            message.save()
            html_body = render_to_string('game_app/email_template.html', data)
            msg = EmailMultiAlternatives(f'Вопрос #{message.pk}', html_body, from_email=settings.EMAIL_HOST_USER,
                                         to=[settings.EMAIL_HOST_USER, ])
            msg.content_subtype = "html"
            msg.send()
            data = {
                'topic': 'Успешно',
                'text':
                    '''Ваш вопрос успешно доставлен<br>Ожидайте ответа''',
                'user': request.user.email,
            }
            html_body = render_to_string('game_app/email_template.html', data)
            msg = EmailMultiAlternatives(f'Вопрос #{message.pk}', html_body, from_email=settings.EMAIL_HOST_USER,
                                         to=[data["user"], ])
            msg.content_subtype = "html"
            msg.send()
            return redirect('profile', request.user.username)
        else:
            return HttpResponse('error')


class SubscribeView(View):
    def post(self, request):
        topics = list(MessageTopics.objects.filter(name__in=request.POST.getlist('topics', '')))
        for topic in MessageTopics.objects.all():
            if topic in topics:
                topic.users.add(request.user)
                topic.save()
            else:
                topic.users.remove(request.user)
                topic.save()

        return redirect('support')


class ChatView(View):
    def get(self, request):
        context = {
            "room_name": "group",
            'messages': ChatMessage.objects.all(),
        }
        return render(request, template_name="game_app/chat.html", context=context)
