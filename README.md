A simple API written in python to translate text from one language to another.

Not necessrely always perfect, but genrally good enough for any reasonable purpose.

The request is to be sent to `localhost:8000/translate` with a body formated as follows:

```json
{
    "text": "text to translate goes here",
    "target_lang": "fr"
}
```

text is the text to be translated, target_lang is the, well, target language.

the returned output will be:

```json
{
    "source_text": "text to translate goes here",
    "source_lang": "en",
    "translated_text": "texte à traduire va ici"
}
```

with source text being the original text, source lang being the language of the original message, and translated text being the text translated to the requested language, english by default.
