import re
import emoji
import enchant
import pandas as pd
from tqdm import tqdm
from ekphrasis.dicts.emoticons import emoticons
from google.transliteration import transliterate_word
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor

enCheck = enchant.Dict("en_US")
tamizh = pd.read_csv('./data.csv')

preprocessor =  TextPreProcessor(
    # terms that will be normalized
    normalize = [
        'url', 
        'email',
        'percent',
        'money',
        'phone',
        'user',
        'time',
        'url',
        'date',
        'number'
        ],
    # terms that will be annotated
    annotate = {
        "hashtag",
        "allcaps",
        "elongated",
        "repeated",
        "emphasis",
        "censored"
        },
    fix_html = False,
    segmenter = "twitter",
    corrector = "twitter",
    unpack_hashtags = True,  
    unpack_contractions = True,  # Unpack contractions (can't -> can not)
    spell_correct_elong = False,  # spell correction for elongated words
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer = SocialTokenizer(lowercase=True).tokenize,
    remove_tags = False,
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts = [emoticons]
    )

tags = {
    "ALLCAPS": "(?<![#@$])\\b([A-Z][A-Z ]{1,}[A-Z])\\b",
    "CENSORED": "(?:\\b\\w+\\*+\\w+\\b)",
    "DATE": "(?:(?:(?:(?:(?<!:)\\b\\'?\\d{1,4},? ?)?\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,4})\\b))|(?:(?:(?<!:)\\b\\'?\\d{1,4},? ?)\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,4})\\b)?))|(?:\\b(?<!\\d\\.)(?:(?:(?:[0123]?[0-9][\\.\\-\\/])?[0123]?[0-9][\\.\\-\\/][12][0-9]{3})|(?:[0123]?[0-9][\\.\\-\\/][0123]?[0-9][\\.\\-\\/][12]?[0-9]{2,3}))(?!\\.\\d)\\b))",
    "ELONGATED": "\\b[A-Za-z]*([a-zA-Z])\\1\\1[A-Za-z]*\\b",
    "EMAIL": "(?:^|(?<=[^\\w@.)]))(?:[\\w+-](?:\\.(?!\\.))?)*?[\\w+-]@(?:\\w-?)*?\\w+(?:\\.(?:[a-z]{2,})){1,3}(?:$|(?=\\b))",
    "EMPHASIS": "(?:\\*\\b\\w+\\b\\*)",
    "HASHTAG": "#",
    "HASHTAG": "\\#\\b[\\w\\-\\_]+\\b",
    "REPEATED": "(.)\\1{2,}",
    "MONEY": "(?:[$\u20ac\u00a3\u00a2]\\d+(?:[\\.,']\\d+)?(?:[MmKkBb](?:n|(?:il(?:lion)?))?)?)|(?:\\d+(?:[\\.,']\\d+)?[$\u20ac\u00a3\u00a2])",
    "NUMBER": "\\b\\d+(?:[\\.,']\\d+)?\\b",
    "PERCENT": "\\b\\d+(?:[\\.,']\\d+)?\\b%",
    "PHONE": "(?<![0-9])(?:\\+\\d{1,2}\\s)?\\(?\\d{3}\\)?[\\s.-]?\\d{3}[\\s.-]?\\d{4}(?![0-9])",
    "TAG": "<[\\/]?\\w+[\\/]?>",
    "TIME": "(?:(?:\\d+)?\\.?\\d+(?:AM|PM|am|pm|a\\.m\\.|p\\.m\\.))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\\.m\\.|p\\.m\\.))?)",
    "URL": "(?:https?:\\/\\/(?:www\\.|(?!www))[^\\s\\.]+\\.[^\\s]{2,}|www\\.[^\\s]+\\.[^\\s]{2,})",
    "USER": "\\@\\w+",
}
annotate = [
    "<hashtag>",
    "<allcaps>",
    "<elongated>",
    "<repeated>",
    "<emphasis>",
    "<censored>",
    "</allcaps>"
]

def transformText(corpus):
    '''
    Converts a given code-mixed corpus into either only English or Tamizh
    Parameters:
        corpus (String) : code-mixed text
    Returns: 
        String: processed text
    Operations: transillerate, translate, spell check, de-emojize, alloting tags, annotations, normalize
    '''
    converted_texts = []
    for sent in tqdm(corpus):   
        sent = " ".join(sent.split())
        reconstructed_sent = ""
        for word in sent.split(" "):
            if emoji.demojize(word) != word:
                reconstructed_sent += f' <{emoji.demojize(word)[1:-1]}>'
                continue
            if enCheck.check(word):
                reconstructed_sent += f' {" ".join(preprocessor.pre_process_doc(word))}'
                continue
            elif word.isalpha():
                if enCheck.check(word) == False:
                    temp = preprocessor.pre_process_doc(word)
                    for spl in temp:
                        if spl in annotate:
                            reconstructed_sent += f' {spl}'  
                        else:
                            if enCheck.check(spl):
                                reconstructed_sent += f' {spl}'
                            elif len(transliterate_word(spl, lang_code="ta")) > 0:
                                reconstructed_sent += " " + transliterate_word(spl, lang_code="ta")[0]
                            else:
                                continue
            else:
                word = re.sub(r'#', '# ', word)
                word = re.sub(r'@', '@ ', word)
                for tag, rex in tags.items():
                    word = re.sub(rex, " " + tag + " ", word)
                for sub in word.split(" "):
                    if '#' in sub:
                        reconstructed_sent += f' <hashtag>'
                    elif '@' in sub:
                        reconstructed_sent += f' <user>'
                    elif sub in tags.keys():
                        reconstructed_sent += f' <{sub.lower()}>'
                    else:
                        if len(sub) > 0 :
                            if enCheck.check(sub):
                                reconstructed_sent += f' {sub.lower()}'
                            elif len(transliterate_word(sub, lang_code="ta")) > 0:
                                reconstructed_sent += f' {transliterate_word(sub, lang_code="ta")[0]}'
        converted_texts.append(reconstructed_sent)
    return converted_texts

step = 50
for i in range(0, tamizh.shape[0], step):
    data = tamizh.iloc[i:i+step, :]
    data['text'] = transformText(data['text'])
    data.to_csv(f"./processed_{i}.csv")
