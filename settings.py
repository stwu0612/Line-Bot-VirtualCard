#!/usr/bin/python3
import os
from linebot import (
    LineBotApi, WebhookHandler
)


# LINE
line_bot_api = LineBotApi("wULi7esbKlga4kO7wNAyn8SUp77Ya83FfGeVpuht0KnkaVSNCNm2U7bAnumf4W6P+0drQlYaFRNEb45E6fGWXWLqNJ33mr4h1VbL0rIFHy2RHklhAtE7WkL7JFymWk7jl8a5inBIf+ZHucLAHUOV6gdB04t89/1O/w1cDnyilFU=")
handler = WebhookHandler("0bf728b28ef75774206b46c5597acb69")

# redis
redis_url = os.getenv('REDISTOGO_URL', 'redis://redistogo:0f069d949cd9d5af4cfefec94c57848d@spinyfin.redistogo.com:11025/')
#redis_url = 'redis://redistogo:0f069d949cd9d5af4cfefec94c57848d@spinyfin.redistogo.com:11025/'

# timer
timer_key_name = 'timer'
timer_interval_sec = 1
