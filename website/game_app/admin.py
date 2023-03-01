from django.contrib import admin
from .models import *


class SearchGameAdmin(admin.ModelAdmin):
    search_fields = ('name',)


class SearchLibraryAdmin(admin.ModelAdmin):
    list_filter = ("user",)


class SearchExtraAdmin(admin.ModelAdmin):
    search_fields = ('game_id',)


class SupportAdmin(admin.ModelAdmin):
    list_display = ("user", "date", "IsAnswered")
    list_filter = ("date", "IsAnswered")
    search_fields = ('user',)


def set_users(self, request, queryset):
    for obj in queryset:
        if not obj.isSend:
            obj.shown_to.set(obj.topic.users.all())
            obj.save()


set_users.short_description = "Установить получателей"


class MessageAdmin(admin.ModelAdmin):
    list_display = ("topic", "send_at", "showned", "isSend")
    list_filter = ("isSend", "send_at",)
    search_fields = ('topic',)
    actions = [set_users, ]

    def showned(self, obj):
        return f"{len(list(obj.shown_to.all()))} человек"

    showned.short_description = 'Увидит/Увидело'


class MessageTopicsAdmin(admin.ModelAdmin):
    list_display = ("name", 'showned')
    search_fields = ('name',)

    def showned(self, obj):
        msg = Message.objects.filter(topic=obj)
        # print(msg)
        sum = 0

        for message in msg:
            if message.isSend:
                sum += len(list(message.shown_to.all()))
        return f"{sum} сообщений"

    showned.short_description = 'Отправлено'


admin.site.register(Reviews)
admin.site.register(Profile)
admin.site.register(Library, SearchLibraryAdmin)
admin.site.register(Genres)
admin.site.register(Platforms)
admin.site.register(Game, SearchGameAdmin)
admin.site.register(Websites, SearchExtraAdmin)
admin.site.register(TechSupport, SupportAdmin)

admin.site.register(MessageTopics, MessageTopicsAdmin)
admin.site.register(Message, MessageAdmin)

admin.site.register(ChatMessage)
