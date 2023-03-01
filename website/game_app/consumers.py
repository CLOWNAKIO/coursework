import json

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.auth import login, get_user

from game_app.models import ChatMessage
import locale

locale.setlocale(locale.LC_ALL, "ru_RU")  # russian


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = "chat_%s" % self.room_name

        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    @database_sync_to_async
    def likes_update(self, pk, type):
        chat = ChatMessage.objects.get(pk=pk)
        if type == 'like':
            if self.user in chat.like.all() and self.user not in chat.dislike.all():
                chat.like.remove(self.user)
            elif self.user not in chat.like.all() and self.user not in chat.dislike.all():
                chat.like.add(self.user)
            elif self.user not in chat.like.all() and self.user in chat.dislike.all():
                chat.like.add(self.user)
                chat.dislike.remove(self.user)
        elif type == 'dislike':
            if self.user in chat.like.all() and self.user not in chat.dislike.all():
                chat.like.remove(self.user)
                chat.dislike.add(self.user)
            elif self.user not in chat.like.all() and self.user not in chat.dislike.all():
                chat.dislike.add(self.user)
            elif self.user not in chat.like.all() and self.user in chat.dislike.all():
                chat.dislike.remove(self.user)
        chat.save()
        return {"like": [i.username for i in chat.like.all()],
                'dislike': [i.username for i in chat.dislike.all()],
                }

    @database_sync_to_async
    def get_likes(self, pk):
        chat = ChatMessage.objects.get(pk=pk)
        return {"like": [i.username for i in chat.like.all()],
                'dislike': [i.username for i in chat.dislike.all()],
                'like_img': [i.profile.avatar.url if i.profile.avatar else "/static/game_app/icons/icons/circle.svg" for
                             i in chat.like.all()],
                "dislike_img": [i.profile.avatar.url if i.profile.avatar else "/static/game_app/icons/icons/circle.svg"
                                for i in chat.dislike.all()],
                }

    @database_sync_to_async
    def set_parent(self, obj, pk):
        parent = ChatMessage.objects.get(pk=pk)
        obj.question = parent
        obj.save()

        return {"parent_pk": obj.question.pk,
                "parent_message": obj.question.content,
                "parent_user": obj.question.author.username, }

    # Receive message from WebSocket
    async def receive(self, text_data):
        data = json.loads(text_data)
        # print(data)
        if data["type"] == "chat_message":
            message = data["message"]

            # save message in db
            chat = ChatMessage(
                author=self.scope['user'],
                content=message,
            )
            await database_sync_to_async(chat.save)()
            parent = {}
            if data["parent"]:
                parent = await self.set_parent(chat, data["parent"])

            res = await self.get_likes(chat.pk)

            response = {"type": "chat_message",
                        "message": message,
                        "user": self.scope['user'].username,
                        "time": chat.date.strftime('%d %B %Y г. %H:%M'),
                        "pk": chat.pk,
                        } | parent | res

            # Send message to room group
            await self.channel_layer.group_send(
                self.room_group_name, response
            )
        if data["type"] == 'like' or data["type"] == 'dislike':
            pk = int(data["message"])
            await self.likes_update(pk, data["type"])
            # print(res)
            res = await self.get_likes(pk)
            response = {"type": "rate",
                        "message": pk,
                        } | res

            await self.channel_layer.group_send(
                self.room_group_name, response)

    # Receive message from room group
    async def chat_message(self, event):
        # Send message to WebSocket
        await self.send(text_data=json.dumps(event))

    async def rate(self, event):
        # Send message to WebSocket
        await self.send(text_data=json.dumps(event))
