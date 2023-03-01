import sqlite3
from datetime import datetime
from requests import post

platforms_arr = {'Android': 1, 'Linux': 2, 'Mac': 3, 'PC (Microsoft Windows)': 4, 'PlayStation': 5}

genres_arr = {'Thriller': 1, 'Science fiction': 2, "Action": 3, "Horror": 4, "Survival": 5, "Fantasy": 6,
              "Historical": 7,
              "Stealth": 8, "Comedy": 9, "Business": 10, "Drama": 11, "Non-fiction": 12, "Kids": 13, "Sandbox": 14,
              "Open world": 15,
              "Warfare": 16, "4X (explore, expand, exploit, and exterminate)": 17, "Educational": 18, "Mystery": 19,
              "Party": 20,
              "Romance": 21, "Erotic": 22, "Fighting": 23, "Shooter": 24, "Music": 25, "Platform": 26, "Puzzle": 27,
              "Racing": 28,
              "Real Time Strategy (RTS)": 29, "Role-playing (RPG)": 30, "Simulator": 31, "Sport": 32, "Strategy": 33,
              "Turn-based strategy (TBS)": 34, "Tactical": 35, "Quiz/Trivia": 36, "Hack and slash/Beat 'em up": 37,
              "Pinball": 38,
              "Adventure": 39, "Arcade": 40, "Visual Novel": 41, "Indie": 42, "Card & Board Game": 43, "MOBA": 44,
              "Point-and-click": 45,
              "Single player": 46, "Multiplayer": 47, "Co-operative": 48, "Split screen": 49,
              "Massively Multiplayer Online (MMO)": 50,
              "Battle Royale": 51, "First person": 52, "Third person": 53, "Bird view / Isometric": 54, "Text": 55,
              "Side view": 56,
              "Virtual Reality": 57, "Auditory": 58}


def init_token():
    client_id = 'l5ogt2dv7937ykpust4nsb9235cbp6'
    secret_code = 'eds88axehl8gw6kur9qkpdb0zk9e4n'

    url = f'https://id.twitch.tv/oauth2/token?client_id={client_id}&client_secret={secret_code}&grant_type=client_credentials'

    request = post(url).json()

    token, expires_in, token_type = request.values()

    _headers = {'Client-ID': client_id, 'Authorization': token_type + ' ' + token}

    return _headers


def post_req(obj: str, req: str, headers=init_token()):
    url = 'https://api.igdb.com/v4/'
    request = post(url + obj, headers=headers, data=req).json()

    return request


def fill_genres():
    db = sqlite3.connect('db.sqlite3')
    cur = db.cursor()
    limit = 500
    req = f'fields id, name; limit {limit};'

    all_genres = [post_req('themes', req), post_req('genres', req), post_req('game_modes', req),
                  post_req('player_perspectives', req)]

    arr = []
    for i in all_genres:
        for j in i:
            arr.append(j['name'])

    for i in arr:
        try:
            cur.execute(f'''insert into game_app_genres (name) values ("{i}")''')
        except:
            print('GENRES error')
    else:
        db.commit()
    db.close()


def update_genres():
    db = sqlite3.connect('db.sqlite3')
    cur = db.cursor()
    cur.execute(f'''update game_app_genres set id = id - 58''')


def fill_platforms():
    out = ['Nintendo', 'Linux', 'Xbox', 'Sega', 'PlayStation', 'PC (Microsoft Windows)', 'iOS', 'Android']
    db = sqlite3.connect('db.sqlite3')
    cur = db.cursor()
    for i in out:
        try:
            cur.execute(f'''insert into game_app_platforms (name) values ("{i}")''')
        except:
            print('PLAT error')
    else:
        db.commit()
        pass
    db.close()


def get_popular():
    limit = 6
    diff = 3944615
    date = datetime.timestamp(datetime.strptime('2022.10.01', '%Y.%m.%d'))
    now = datetime.timestamp(datetime.now())

    req = f'fields id, name, platforms.name, cover.image_id, genres.name, game_modes.name, themes.name; limit {limit}; ' \
          f'where first_release_date > {int(now - diff)} & total_rating_count > 0; sort total_rating_count desc;'

    popular = post_req('games', req)
    for i in popular:
        print(i)


def get_rec_released():
    limit = 6
    now = datetime.timestamp(datetime.now())
    req = f'fields id, name, first_release_date; ' \
          f'limit {limit}; where first_release_date < {int(now)}; sort first_release_date desc; '

    recently_released = post_req('games', req)

    for i in recently_released:
        print(i)


def get_games(offset, limit=500):
    req = f'fields id, name, platforms.name, cover.image_id, genres.name, game_modes.name, themes.name, summary, ' \
          f'storyline, player_perspectives.name, first_release_date, age_ratings.rating, ' \
          f'involved_companies.company.name; limit {limit}; offset {offset}; where id > 0; sort id; '

    games = post_req('games', req)
    # print(games)
    res = []
    for game in games:
        genres = []
        platforms = []
        id_game = 0
        name = ''
        desc = ''
        story = ''
        image_id = 0
        release_date = ''
        age_rating = 0
        developer = ''

        id_game = game['id']
        name = game['name']
        try:
            for z in game['platforms']:
                platforms.append(z['name'])
        except:
            pass
        try:
            for z in game['game_modes']:
                genres.append(z['name'])
        except:
            pass
        try:
            for z in game['genres']:
                genres.append(z['name'])
        except:
            pass
        try:
            for z in game['game_modes']:
                genres.append(z['name'])
        except:
            pass
        try:
            for z in game['themes']:
                genres.append(z['name'])
        except:
            pass

        try:
            for z in game['player_perspectives']:
                genres.append(z['name'])
        except:
            pass

        try:
            image_id = game['cover']['image_id']
        except:
            pass

        try:
            release_date = datetime.strftime(datetime.fromtimestamp(game['first_release_date']), '%Y-%m-%d')
        except:
            pass

        try:
            age_rating = game['age_ratings'][0]['rating']
        except:
            pass

        try:
            for z in game['involved_companies']:
                developer = z['company']['name']
        except:
            pass

        try:
            desc += game['summary']
        except:
            pass

        try:
            story += game['storyline']
        except:
            pass

        # res.append({'id': id_game, 'name': name, 'genres': genres, 'plats': platforms, 'image': image_id,
        #             'release': release_date, 'age': age_rating, 'dev': developer, 'desc': desc + '\n' + story})
        res.append([id_game, name, release_date, developer, age_rating, desc + '\n' + story, genres, platforms])
        # print(res[-1]['id'], end=' ')
    return res


def fill_games():
    db = sqlite3.connect('db.sqlite3')
    cur = db.cursor()
    offset = 0
    while offset < 300_000:
        try:
            games = get_games(offset)
            # print(games)
            print('iter', offset // 500)
            for game in games:
                for genre in game[-2]:
                    for platform in game[-1]:
                        fill = game[:-2]
                        fill.append(genres_arr[genre])
                        fill.append(platforms_arr[platform])
                        print(fill)
                        # cur.execute(f'''insert into game_app_game (game_id, name, release_date, developer, age_rating,
                        # description, genres_id, platforms_id) values ({game['id']}, "{game['name']}", date("{game['release']}"),
                        #  "{game['dev']}", {game['age']}, "{game['desc']}", {genre_id}, {plat_id})''')
                        cur.executemany(
                            """INSERT INTO game_app_game (game_id, name, release_date, developer, age_rating, description, genres_id, platforms_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                            fill)
                        # print(cur.execute("select name from game_app_game").fetchone())

        except:
            print('error')
            # print(cur.connection)
        else:
            # db.commit()
            offset += 500

    db.close()


def get_game(offset, limit = 500):
    out = ['PC (Microsoft Windows)', 'iOS', 'Android']
    web_cats = ['', 'official', '', 'wikia', 'wikipedia', 'facebook', 'twitter', 'twitch', 'instagram', 'youtube',
                'iphone',
                'ipad', 'android', 'steam', 'reddit', 'itch', 'epicgames', 'gog', 'discord']
    cor_web_cats = [1, 13, 16, 17]

    # print(f'({", ".join(out)})')
    req = f'fields id, name, slug, similar_games.id, artworks.image_id, videos.video_id, screenshots.image_id, ' \
          f'first_release_date, release_dates.date,release_dates.platform.name, release_dates.date,' \
          f'release_dates.platform.platform_family.name, genres.name, game_modes.name, themes.name, ' \
          f'player_perspectives.name, platforms.name, platforms.platform_family.name, total_rating, ' \
          f'total_rating_count, websites.url, websites.category, cover.image_id, summary, storyline, ' \
          f'involved_companies.company.name; limit {limit}; where ' \
          f'(platforms.name = "PC ("* | platforms.name = "IOS" | platforms.name = "Android" | ' \
          f'platforms.platform_family != null) & id > {offset} & release_dates != null & involved_companies != null; ' \
          f'sort id;'
    game = post_req('games', req)
    # for i in game:
    #     for j in i:
    #         print(j, " : ", i[j])
    #     print()
    # req = 'fields name; where name = "Android"; limit 500; '
    # game = post_req('platforms', req)

    output = []
    for i in game:
        # print(i["id"])
        fill = dict()
        fill["game_id"] = i["id"]
        fill["name"] = i["name"]
        fill["slug"] = i["slug"]
        if "cover" in i:
            fill["cover"] = i["cover"]["image_id"]
        else:
            fill["cover"] = ''
        if "first_release_date" in i:
            # print(i['first_release_date'])
            fill["release_date"] = datetime.strftime(datetime.fromtimestamp(abs(i['first_release_date'])), '%Y-%m-%d')
        else:
            fill["release_date"] = ''
        fill["release_dates"] = {}
        if "release_dates" in i:
            for z in i["release_dates"]:
                if 'platform' in z:
                    if "platform_family" in z["platform"]:
                        fill['release_dates'][z["platform"]['platform_family']['name']] = datetime.strftime(
                            datetime.fromtimestamp(abs(z["date"])), '%Y-%m-%d') if "date" in z else ''
                    else:
                        fill['release_dates'][z["platform"]['name']] = datetime.strftime(datetime.fromtimestamp(abs(z["date"])),
                                                                                     '%Y-%m-%d') if "date" in z else ''
        if "involved_companies" in i:
            for z in i["involved_companies"]:
                fill["developer"] = z['company']['name']
        else:
            fill["developer"] = ''
        fill['websites'] = {}
        if 'websites' in i:
            for z in i['websites']:
                if 'category' in z:
                    if z['category'] in cor_web_cats:
                        fill['websites'][web_cats[z['category']]] = z['url']
        fill['genres'] = []
        if "genres" in i:
            for z in i['genres']:
                fill['genres'].append(z['name'])
        if "game_modes" in i:
            for z in i['game_modes']:
                fill['genres'].append(z['name'])
        if "player_perspectives" in i:
            for z in i['player_perspectives']:
                fill['genres'].append(z['name'])
        if "themes" in i:
            for z in i['themes']:
                fill['genres'].append(z['name'])
        fill['platforms'] = []
        for z in i['platforms']:
            if "platform_family" in z:
                fill['platforms'].append(z['platform_family']['name'])
            else:
                fill['platforms'].append(z['name'])
        if "total_rating" in i:
            fill["rating"] = i['total_rating']
        else:
            fill["rating"] = 0
        if "total_rating_count" in i:
            fill["rating_count"] = i["total_rating_count"]
        else:
            fill["rating_count"] = 0
        fill["description"] = ''
        if "summary" in i:
            fill["description"] += i["summary"]
        if "storyline" in i:
            fill["description"] += i["storyline"]
        output.append(fill)
        fill["images"] = []
        if "artworks" in i:
            for z in i["artworks"]:
                fill["images"].append(z["image_id"])
        if "screenshots" in i:
            for z in i["screenshots"]:
                fill["images"].append(z["image_id"])
        fill["video"] = []
        if "videos" in i:
            for z in i["videos"]:
                fill["video"].append(z["video_id"])

    # for i in output:
    #     for j in i:
    #         print(j, ':', i[j])
    #
    #     print('-' * 30)

    return output

def get_similar(id):
    req = f'fields similar_games.id; where id = {id}; sort similar_games.id;'
    game = post_req('games', req)
    ID = []
    for ids in game[0]["similar_games"]:
        ID.append(ids["id"])
    return ID

if __name__ == '__main__':
    # fill_genres()
    # fill_platforms()
    # fill_games()
    # get_game(26750, 100)
    get_similar(2963)
    # print(post_req('games', 'fields release_dates; where slug = "death-stranding-2";'))
    pass
