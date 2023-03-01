from django.core.mail import EmailMultiAlternatives
from django.db import models
from django.db.models.signals import post_save, pre_save, post_init
from django.dispatch import receiver
from django.template.loader import render_to_string
from django.utils import timezone
from django.contrib.auth.models import User
from django.conf import settings


class Genres(models.Model):
    name = models.CharField(max_length=30, db_index=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'жанр'
        verbose_name_plural = 'жанры'


class Platforms(models.Model):
    name = models.CharField(max_length=30, db_index=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'платформа'
        verbose_name_plural = 'платформы'


class Game(models.Model):
    game_id = models.IntegerField(primary_key=True, db_index=True)
    name = models.CharField(max_length=200)
    genres = models.ManyToManyField(Genres)
    platforms = models.ManyToManyField(Platforms)
    release_date = models.DateField(null=True)
    developer = models.CharField(max_length=50)
    description = models.TextField(blank=True)
    cover = models.CharField(null=True, max_length=200)
    slug = models.CharField(null=False, max_length=200)
    rating = models.PositiveSmallIntegerField(null=True)
    rating_count = models.PositiveIntegerField(null=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'игра'
        verbose_name_plural = 'игры'


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True, db_index=True)
    avatar = models.ImageField(upload_to="photos/%Y/%m/%d/", null=True, blank=True)
    nickname = models.CharField(max_length=30, blank=False)

    def __str__(self):
        return self.nickname

    class Meta:
        verbose_name = 'профиль'
        verbose_name_plural = 'профили'


class Reviews(models.Model):
    text = models.TextField(blank=True)
    date = models.DateTimeField(auto_now=True)
    game = models.ForeignKey(Game, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"Отзыв на {self.game.name} пользователем {self.user.username}"

    class Meta:
        verbose_name = 'отзыв'
        verbose_name_plural = 'отзывы'


class Library(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    game = models.ForeignKey(Game, on_delete=models.CASCADE)
    review = models.ForeignKey(Reviews, on_delete=models.CASCADE, null=True, blank=True)
    rate = models.PositiveSmallIntegerField(null=True, blank=True)
    added_at = models.DateField(null=False, auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}: {self.game.name}"

    class Meta:
        verbose_name = 'библиотека'
        verbose_name_plural = 'библиотеки'


class ReviewSources(models.Model):
    name = models.CharField(max_length=20, db_index=True)
    link = models.URLField()

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'источник'
        verbose_name_plural = 'источники'


class Websites(models.Model):
    game_id = models.IntegerField()
    name = models.CharField(max_length=30, null=False)
    url = models.CharField(max_length=200, null=False)

    class Meta:
        verbose_name = 'сайт'
        verbose_name_plural = 'сайты'

    def __str__(self):
        return str(self.game_id) + ' ' + self.name


class ReleaseDates(models.Model):
    game_id = models.IntegerField()
    platform = models.CharField(max_length=30, null=False)
    date = models.DateField(null=True)

    class Meta:
        verbose_name = 'Дата выхода'
        verbose_name_plural = 'Даты выхода'

    def __str__(self):
        return str(self.game_id) + ' ' + self.platform


class Images(models.Model):
    game_id = models.IntegerField()
    image_id = models.CharField(primary_key=True, null=False, max_length=200)

    class Meta:
        verbose_name = 'Картинки'
        verbose_name_plural = 'Картинки'

    def __str__(self):
        return self.image_id


class Videos(models.Model):
    game_id = models.IntegerField()
    video_id = models.CharField(primary_key=True, null=False, max_length=200)

    class Meta:
        verbose_name = 'Видео'
        verbose_name_plural = 'Видео'

    def __str__(self):
        return str(self.game_id) + ' ' + self.video_id


class TechSupport(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(null=False, auto_now_add=True, editable=False)
    question = models.TextField(null=False, blank=False)
    answer = models.TextField(blank=True)
    IsAnswered = models.BooleanField(blank=True, default=False)

    class Meta:
        verbose_name = 'Техническая поддержка'
        verbose_name_plural = 'Технические поддержки'

    def __str__(self):
        return self.user.username


class MessageTopics(models.Model):
    name = models.CharField(max_length=250)
    users = models.ManyToManyField(User, blank=True, verbose_name='Подписчики')

    class Meta:
        verbose_name = 'Тема рассылки'
        verbose_name_plural = 'Темы для рассылок'

    def __str__(self):
        return self.name


class Message(models.Model):
    topic = models.ForeignKey(MessageTopics, on_delete=models.SET_DEFAULT, default='Topic was deleted')
    text = models.TextField()
    send_at = models.DateTimeField(auto_now=True)
    shown_to = models.ManyToManyField(User, blank=True, editable=False)
    isSend = models.BooleanField(blank=True, default=False)

    class Meta:
        verbose_name = 'Рассылка'
        verbose_name_plural = 'Рассылки'

    def __str__(self):
        return self.topic.name


@receiver(post_save, sender=User)
def update_profile_signal(sender, instance, created, **kwargs):
    if created:
        profile = Profile.objects.create(user=instance)
        profile.nickname = instance.username
        instance.profile.save()


@receiver(post_save, sender=TechSupport)
def send_answer(sender, instance, created, **kwargs):
    if instance.IsAnswered:
        data = {
            'topic': f'Ответ на вопрос №{instance.pk}',
            'text': instance.answer,
            'user': 'Техническая поддержка сайта',
        }
        html_body = render_to_string('game_app/email_template.html', data)
        msg = EmailMultiAlternatives(data['topic'], html_body, from_email=settings.EMAIL_HOST_USER,
                                     to=[instance.user.email, ])
        msg.content_subtype = "html"
        msg.send()


@receiver(post_save, sender=MessageTopics)
def set_users_for_message(sender, instance, **kwargs):
    messages = Message.objects.filter(topic=instance)
    for message in messages:
        if not message.isSend:
            message.shown_to.set(instance.users.all())
            message.save()


@receiver(post_save, sender=Message)
def send_message(sender, instance, created, **kwargs):
    # instance.shown_to.set(instance.topic.users.all())
    # print( instance.shown_to.all())
    if instance.isSend:
        data = {
            'topic': instance.topic.name,
            'text': instance.text,
            'user': 'Техническая поддержка сайта',
        }
        emails = [user.email for user in instance.topic.users.all()]
        html_body = render_to_string('game_app/email_template.html', data)
        msg = EmailMultiAlternatives(data['topic'], html_body, from_email=settings.EMAIL_HOST_USER,
                                     to=emails)
        msg.content_subtype = "html"
        msg.send()


class ChatMessage(models.Model):
    question = models.ForeignKey("ChatMessage", on_delete=models.CASCADE, default=None, null=True, related_name='parent')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comment_author')
    content = models.TextField()
    date = models.DateTimeField(auto_now_add=True, editable=False)
    like = models.ManyToManyField(User, related_name='comment_like')
    dislike = models.ManyToManyField(User, related_name='comment_dislike')

    def get_prev_messages(self, timestamp, n=100):
        return ChatMessage.objects.filter(date__lte=timestamp)[:n]

    class Meta:
        verbose_name = 'Сообщениe чата'
        verbose_name_plural = 'Сообщения чата'
