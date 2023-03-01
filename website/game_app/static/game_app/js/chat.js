        const messageInputDom = document.querySelector('#chat-message-input');
        const footer = document.querySelector('footer');
        const loggedInUser = JSON.parse(document.getElementById('user_id').textContent)

        window.onload = function()
         {
            window.scrollTo( 0, footer.offsetTop )
         }

        const chatLog = document.querySelector('#chat-log')
        const roomName = JSON.parse(document.getElementById('room-name').textContent);

        const chatSocket = new WebSocket(
            'ws://'
            + window.location.host
            + '/ws/chat/'
            + roomName
            + '/'
        );

        function newElement(tag, className, Text = '')
        {
            el = document.createElement(tag);
            if (className)
                        el.classList.add(className);
            el.innerText = Text;

            return el;
        }

        chatSocket.onmessage = function(e) {
            const data = JSON.parse(e.data);

            if(data.type === 'chat_message')
            {
                const messageElement = document.createElement('div')
                const user = data.user

                if (user === loggedInUser) {
                    messageElement.classList.add('message', 'sender')
                } else {
                    messageElement.classList.add('message', 'receiver')
                }

                messageElement.dataset.pk = data.pk;
                chatLog.appendChild(messageElement)

                const user_field = newElement('div', 'user-name')
                const user_name = newElement('span', '', user)

                time = data.time
                if (time[0] === '0')
                    delete time[0]
                time = time.toLowerCase();

                const date = newElement('span', '', time)
                user_field.appendChild(user_name)
                user_field.appendChild(date)
                messageElement.appendChild(user_field)

                if (data.parent_user)
                {
                        const parent_pk = data.parent_pk;

                        const parent = newElement('div', 'parent-message');
                        parent.dataset.pk = data.parent_pk;

                        const parent_user = newElement('div', 'user-name', data.parent_user);
                        const parent_message = newElement('div', 'message-content', data.parent_message);

                        parent.appendChild(parent_user);
                        parent.appendChild(parent_message);

                        messageElement.appendChild(parent)
                        parent.onclick = (e) =>
                        {
                            const msg = chatLog.querySelector('.message[data-pk="'+ data.parent_pk +'"]');
                            window.scrollTo(0, msg.offsetTop);
                            msg.classList.add('retrieved');
                            setTimeout(() => {msg.classList.remove('retrieved')}, 800);
                        }
                }

                const message_content = newElement('div', 'message-content', data.message)
                messageElement.appendChild(message_content)

                const rate = newElement('div', 'actions')

                const like = newElement('div', 'like')
                like.onclick = (e) => {SendLike(e)}

                const dislike = newElement('div', 'dislike')
                dislike.onclick = (e) => {SendDisLike(e)}

                 const like_users = newElement('div', 'like-users')
                 const dislike_users = newElement('div', 'dislike-users')

                rate.appendChild(like);
                rate.appendChild(like_users);
                rate.appendChild(dislike);
                rate.appendChild(dislike_users);


                if (loggedInUser)
                {
                    const reply = newElement('div', 'reply', 'Ответить')
                    reply.onclick = document.querySelector('.reply').onclick;
                    rate.appendChild(reply);
                }

                messageElement.appendChild(rate);

                const img_like = newElement('span', 'material-symbols-outlined', "thumb_up")
                const like_count = newElement('span', '', data.like.length)

                like.appendChild(img_like);
                like.appendChild(like_count);



                const img_dislike = newElement('span', 'material-symbols-outlined', "thumb_down")
                const dislike_count = newElement('span', '', data.dislike.length)

                dislike.appendChild(img_dislike);
                dislike.appendChild(dislike_count);

                 if (document.querySelector('#emptyText'))
                 {
                    document.querySelector('#emptyText').remove()
                 }


             };

             if(data.type === 'rate')
            {
                const id = data.message
                const msg = document.querySelector('.message[data-pk="'+id+'"]')
                const like = msg.querySelector('.like')
                const dislike = msg.querySelector('.dislike')
                like.children[1].innerText = data.like.length
                dislike.children[1].innerText = data.dislike.length

                msg.querySelector('.like-users').remove()
                msg.querySelector('.dislike-users').remove()

                const like_users = newElement('div', 'like-users')
                const dislike_users = newElement('div', 'dislike-users')

                like.after(like_users)
                dislike.after(dislike_users)
//                console.log(data.like, data.dislike)
//                console.log(data.like_img, data.dislike_img)
                for(let i = 0; i < data.like.length; i += 1)
                 {
                    const img = document.createElement('img')
                    img.src = data.like_img[i]
                    img.title = data.like[i]
                    like_users.appendChild(img)
                 }

                 for(let i = 0; i < data.dislike.length; i += 1)
                 {
                    const img = document.createElement('img')
                    img.src = data.dislike_img[i]
                    img.title = data.dislike[i]
                    dislike_users.appendChild(img)
                 }

                if (data.dislike.includes(loggedInUser))
                    dislike.children[0].dataset.fill = 'true';
                else
                    dislike.children[0].dataset.fill = 'false';

                if (data.like.includes(loggedInUser))
                    like.children[0].dataset.fill = 'true';
                else
                    like.children[0].dataset.fill = 'false';
            };
        };

        chatSocket.onclose = function(e) {
            console.error('Chat socket closed unexpectedly');
        };

        if (loggedInUser)
        {
        document.querySelector('#chat-message-input').focus();
        document.querySelector('#chat-message-input').onkeyup = function(e) {
            if (e.keyCode === 13 && !e.shiftKey) {  // enter, return
                document.querySelector('#chat-message-submit').click();
            }
        };

        document.querySelector('#chat-message-submit').onclick = function(e) {
            const message = messageInputDom.innerText;
            if (message.replace(/\n/g,''))
            {
                const parent = document.querySelector('.send-container .parent-message')
                const parent_id = parent.dataset.pk;
                chatSocket.send(JSON.stringify({
                    'type': "chat_message",
                    'message': message,
                    'parent': parent_id ? parent_id : null,
                }));
                messageInputDom.innerText = '';
                document.querySelector('#cancel').click();
            }
        };
        }

        function SendLike(e)
        {
         if (loggedInUser)
            chatSocket.send(JSON.stringify({
                    'type': "like",
                    'message': e.target.closest('.message').dataset.pk,
                }));
        }

        function SendDisLike(e)
        {
         if (loggedInUser)
            chatSocket.send(JSON.stringify({
                    'type': "dislike",
                    'message': e.target.closest('.message').dataset.pk,
                }));
        }

        document.querySelectorAll('.like').forEach(el => el.onclick = (e) => {SendLike(e)})

        document.querySelectorAll('.dislike').forEach(el => el.onclick = (e) => {SendDisLike(e)})

        document.querySelectorAll('.reply').forEach(el =>
            el.onclick = (e) => {

            const msg = e.target.closest('.message');
            const parent_user = msg.children[0].children[0];
            const parent_message = msg.querySelector(':scope > .message-content')

            const container = document.querySelector('.send-container');
            delete container.children[0].dataset.pk;
            const user_field = container.children[0].children[0];
            const message_field = container.children[0].children[1];

            user_field.innerText = parent_user.innerText;
            message_field.innerText = parent_message.innerText;

            container.children[0].classList.remove('none');
            container.children[0].dataset.pk = msg.dataset.pk;

            const message = container.children[1];
        })

         if (loggedInUser)
            document.querySelector('#cancel').onclick =
                (e) => {
            const container = document.querySelector('.send-container');
            container.children[0].classList.add('none');
            delete container.children[0].dataset.pk;
        }

        document.querySelectorAll('.parent-message').forEach(el =>
            el.onclick = (e) => {
            const msg = chatLog.querySelector('.message[data-pk="'+ e.target.closest('.parent-message').dataset.pk +'"]');
            window.scrollTo(0, msg.offsetTop);
            msg.classList.add('retrieved');
            setTimeout(() => {msg.classList.remove('retrieved')}, 800);
        });

        document.querySelectorAll('.like-users').forEach(el =>
        {
            if (!el.children.length)
            {
                el.innerText = '';
            }
        })

                document.querySelectorAll('.dislike-users').forEach(el =>
        {
            if (!el.children.length)
            {
                el.innerText = '';
            }
        })

