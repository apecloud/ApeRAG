import ssl

import requests
from bs4 import BeautifulSoup

ssl._create_default_https_context = ssl._create_unverified_context

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'
}


def get_page(url):
    try:
        r = requests.get(url, headers=headers, timeout=20)
        return BeautifulSoup(r.text, 'html.parser')
    except Exception:
        print('requirement fail', url)
        return None


def parse(soup):
    data = {}

    # Title
    data['title'] = soup.select('.question-hyperlink')[0].text

    # Question
    question = soup.select('#question')[0]
    data['question'] = question.select('.s-prose')[0].get_text()
    data['question_comments'] = [c.get_text() for c in question.select('.comment-copy')]

    # Answer
    data['answers'] = []
    for answer in soup.select('#answers .answer'):
        ans = {}
        ans['content'] = answer.select('.s-prose')[0].get_text()
        ans['comments'] = [c.get_text() for c in answer.select('.comment-copy')]
        data['answers'].append(ans)

    return data


def get_text(data, url):
    result = ''

    result += url + '\n'
    result += 'Title: ' + data['title'] + '\n'
    result += 'Question : \n' + data['question'] + '\n'

    if data['question_comments']:
        result += 'Question Comment: \n'
        for c in data['question_comments']:
            result += c + '\n'

    result += 'Answer : \n'
    for i, ans in enumerate(data['answers']):
        result += str(i + 1) + '. \n' + ans['content'] + '\n'

        if ans['comments']:
            result += 'Answer Comment: \n'
            for c in ans['comments']:
                result += c + '\n'

        result += '\n'

    return result


def get_stackoverflow(url):
    soup = get_page(url)
    if soup:
        data = parse(soup)
        return get_text(data, url)
