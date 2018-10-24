# Sombrero

Sombrero is a content extraction project in webpages. It is based on a the
Ricker wavelet also called the Mexican hat wavelet.

It is purely based on signal processing and requires no linguistic knowledge
hence it should work in any language.

# Demo
From the `sombrero/` folder, run the following command in a shell.

```
python3 -m scripts.run_detection --url=https://www.lemonde.fr/proche-orient/article/2018/10/24/l-affaire-khashoggi-nuit-tellement-a-l-image-de-l-arabie-saoudite-qu-elle-laissera-des-traces_5374039_3218.html
```

It prints a list of paragraphs along with their weights. The higher the weight,
the more we believe the paragraph is important for the document.

Simply change the url to fetch any url.
