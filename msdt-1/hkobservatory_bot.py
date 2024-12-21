import json
from typing import Dict, List, Union, Optional

import feedparser
import bs4
import telegram
import telegram.ext

from token import bot_token
from links import weather1, weather2, weather3, weather4, weather5, weather6
from paths import lang, feed, subs


updater = telegram.ext.Updater(token=bot_token)
dispatcher = updater.dispatcher
job_queue = updater.job_queue


def check_feed_update() -> Dict[str, List[feedparser.FeedParserDict]]:
    """
    Checks for updates in the weather feeds and returns any new updates.
    """
    try:
        with open(feed) as f:
            feeds = json.load(f)
        updates = {}
        current_en_update = feedparser.parse(weather1)
        warning_en_update = feedparser.parse(weather2)

        if current_en_update:
            current_en = feeds["current"][0]
            if (
                current_en["entries"][0]["published"]
                != current_en_update.entries[0].published
            ):
                current_trad_update = feedparser.parse(weather3)
                current_simp_update = feedparser.parse(weather4)
                current_update = [
                    current_en_update,
                    current_trad_update,
                    current_simp_update,
                ]
                updates["current"] = current_update
                feeds["current"] = current_update

        if warning_en_update:
            warning_en = feeds["warning"][0]
            if (
                warning_en["entries"][0]["published"]
                != warning_en_update.entries[0].published
            ):
                warning_trad_update = feedparser.parse(weather5)
                warning_simp_update = feedparser.parse(weather6)
                warning_update = [
                    warning_en_update,
                    warning_trad_update,
                    warning_simp_update,
                ]
                updates["warning"] = warning_update
                feeds["warning"] = warning_update

    except FileNotFoundError:
        current_en = feedparser.parse(weather1)
        current_trad = feedparser.parse(weather3)
        current_simp = feedparser.parse(weather4)
        current = [current_en, current_trad, current_simp]

        warning_en = feedparser.parse(weather2)
        warning_trad = feedparser.parse(weather5)
        warning_simp = feedparser.parse(weather6)
        warning = [warning_en, warning_trad, warning_simp]

        with open(feed, "w") as f:
            updates = {"current": current, "warning": warning}
            feeds = updates
            json.dump(updates, f)

    if updates:
        with open(feed, "w") as f:
            json.dump(feeds, f)
    return updates


def get_user_language() -> Dict[str, str]:
    """
    Retrieves the language preferences of all users.
    """
    try:
        with open(lang) as f:
            user_language = json.load(f)
    except FileNotFoundError:
        user_language = {}
    return user_language


def get_topics() -> str:
    """
    Provides a list of available weather topics.
    """
    topics = [
        "Current - Current weather information",
        "Warning - Warnings in force",
    ]
    topics = "The topics I can tell you about are:\n" + "\n".join(topics)
    return topics


def get_feed_message(user_id: str, topic: str) -> str:
    """
    Generates a message with weather information
     based on the user's language preference.
    """
    check_feed_update()
    user_language = get_user_language()
    language = user_language.get(user_id, "english")
    with open(feed) as f:
        feeds = json.load(f)

    if language == "english":
        feed = feeds[topic][0]
    elif language == "traditional":
        feed = feeds[topic][1]
    elif language == "simplified":
        feed = feeds[topic][2]

    format = bs4.BeautifulSoup(feed["entries"][0]["summary"], "html.parser")
    if topic == "current":
        for br in format.find_all("br"):
            if br.previous_element != br:
                br.previous_element.wrap(format.new_tag("p"))
            br.decompose()
        for tr in format.find_all("tr"):
            tr.decompose()
        for span in format.find_all("span"):
            span.decompose()
        for table in format.find_all("table"):
            if table.find_previous("p") != format.p:
                table.find_previous("p").decompose()
            table.decompose()
        message = []
        for string in format.stripped_strings:
            message.append(" ".join(string.split()))
        message = "\n".join(message)

    elif topic == "warning":
        message = format.get_text()
    return message


def start(bot: telegram.Bot, update: telegram.Update) -> None:
    """
    Handles the /start command to introduce the bot.
    """
    message = "Hi, I'm HKObservatoryBot! \
        Type @hkobservatory_bot to see what I can do!"
    bot.sendMessage(chat_id=update.message.chat_id, text=message)


def inline_query(bot: telegram.Bot, update: telegram.Update) -> None:
    """
    Handles inline queries and provides relevant results.
    """
    query = update.inline_query.query
    results = []
    user_id = str(update.inline_query.from_user.id)
    first_name = update.inline_query.from_user.first_name

    if not query:
        results.append(
            telegram.InlineQueryResultArticle(
                id="commands",
                title="Commands",
                input_message_content=telegram.InputTextMessageContent(
                    (
                        "Type @hkobservatory_bot + one of the following:\n"
                        "topics;\ntellme + topic;\nsubscribe + topic\
                            ;\nunsubscribe + topic;\nenglish;\n繁體中文;\n简体中文;"
                    )
                ),
                description="List of available commands",
            )
        )
    else:
        if query.lower() in "topics":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="topics",
                    title="Topics",
                    input_message_content=telegram.InputTextMessageContent(
                        get_topics()
                    ),
                    description="List of available topics",
                )
            )
        if query.lower() in "tellme current":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="tellme_current",
                    title="Current Weather",
                    input_message_content=telegram.InputTextMessageContent(
                        get_feed_message(user_id, "current")
                    ),
                    description="Current weather from the HK Observatory",
                )
            )
        if query.lower() in "tellme warning":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="tellme_warning",
                    title="Warning",
                    input_message_content=telegram.InputTextMessageContent(
                        get_feed_message(user_id, "warning")
                    ),
                    description="Warnings in force",
                )
            )
        if query.lower() in "subscribe current":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="sub_current",
                    title="Subscribe Current",
                    input_message_content=telegram.InputTextMessageContent(
                        first_name + " has subscribed to: Current"
                    ),
                    description="Subscribe to current to receive updates",
                )
            )
        if query.lower() in "subscribe warning":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="sub_warning",
                    title="Subscribe Warning",
                    input_message_content=telegram.InputTextMessageContent(
                        first_name + " has subscribed to: Warning"
                    ),
                    description="Subscribe to warning to receive updates",
                )
            )
        if query.lower() in "unsubscribe current":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="unsub_current",
                    title="Unsubscribe Current",
                    input_message_content=telegram.InputTextMessageContent(
                        first_name + " has unsubscribed from: Current"
                    ),
                    description="Unsubscribe from current\
                          to stop receiving updates",
                )
            )
        if query.lower() in "unsubscribe warning":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="unsub_warning",
                    title="Unsubscribe Warning",
                    input_message_content=telegram.InputTextMessageContent(
                        first_name + " has unsubscribed from: Warning"
                    ),
                    description="Unsubscribe from warning\
                          to stop receiving updates",
                )
            )
        if query.lower() in "english":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="lang_english",
                    title="English",
                    input_message_content=telegram.InputTextMessageContent(
                        first_name + "'s language changed to English"
                    ),
                    description="Select English as topic information language",
                )
            )
        if query in "繁體中文":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="lang_traditional",
                    title="繁體中文",
                    input_message_content=telegram.InputTextMessageContent(
                        first_name + "'s language changed to 繁體中文"
                    ),
                    description="Select 繁體中文 as topic information language",
                )
            )
        if query in "简体中文":
            results.append(
                telegram.InlineQueryResultArticle(
                    id="lang_simplified",
                    title="简体中文",
                    input_message_content=telegram.InputTextMessageContent(
                        first_name + "'s language changed to 简体中文"
                    ),
                    description="Select 简体中文 as topic information language",
                )
            )
    bot.answerInlineQuery(update.inline_query.id, results, cache_time=0)


def inline_result(bot: telegram.Bot, update: telegram.Update) -> None:
    """
    Processes chosen inline results 
    and updates user subscriptions or preferences.
    """
    result_id = update.chosen_inline_result.result_id
    user_id = str(update.chosen_inline_result.from_user.id)

    if "lang" in result_id:
        language = result_id[5:]
        try:
            with open(lang) as f:
                user_language = json.load(f)
        except FileNotFoundError:
            user_language = {}

        with open(lang, "w") as f:
            if result_id == ("lang_" + language):
                user_language[user_id] = language
            json.dump(user_language, f)

    elif "sub" in result_id:
        topic = result_id[4:]
        try:
            with open(subs) as f:
                subscribers = json.load(f)
        except FileNotFoundError:
            subscribers = {}

        with open(subs, "w") as f:
            if result_id == ("sub_" + topic):
                try:
                    if user_id not in subscribers[topic]:
                        subscribers[topic].append(user_id)
                except KeyError:
                    subscribers[topic] = [user_id]
            elif result_id == ("unsub_" + topic):
                try:
                    subscribers[topic].remove(user_id)
                except:
                    pass
            json.dump(subscribers, f)


def send_update(bot: telegram.Bot, job: telegram.ext.Job) -> None:
    """
    Sends updates to subscribed users if new weather information is available.
    """
    try:
        with open(subs) as f:
            subscribers = json.load(f)
        user_language = get_user_language()
    except FileNotFoundError:
        subscribers = {}

    if subscribers:
        updates = check_feed_update()
        if updates:
            for topic in updates:
                try:
                    for user_id in subscribers[topic]:
                        language = user_language.get(user_id, "english")
                        message = get_feed_message(updates, topic, language)
                        bot.sendMessage(chat_id=user_id, text=message)
                except telegram.Unauthorized:
                    subscribers[topic].remove(user_id)
                except:
                    pass


job_queue.put(telegram.ext.Job(send_update, 3600.0))

start_handler = telegram.ext.CommandHandler("start", start)
dispatcher.add_handler(start_handler)

inline_query_handler = telegram.ext.InlineQueryHandler(inline_query)
dispatcher.add_handler(inline_query_handler)

inline_result_handler = telegram.ext.ChosenInlineResultHandler(inline_result)
dispatcher.add_handler(inline_result_handler)

updater.start_polling()
updater.idle()
