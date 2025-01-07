# GovDataKeywords
A script for downloading all the keywords from GovData and plotting them in a 2D space using word embedding

![GovData keywords in a 2D space](https://github.com/user-attachments/assets/6604530a-53b8-4bb8-9cfc-86756e499ee7)
*All GovData keywords in an interactive map. If you want to click around, download "Keywords.html"*

![GovData Datasets in a 2D space](https://github.com/user-attachments/assets/217ae235-6ddb-4d5b-bd73-425507ed12f8)
*All GovData Datasets that have keywords in an interactive map. If you want to click around, download "Datasets.html" (more to come on this, I've only just finished it)*


## Overview
This python script downloads ALL keywords from all datasets from GovData.de from the beginning until 20.09.2023 that have a title and an identifier. I sorted those into various csv files, most importantly into corpus.csv, which lists all the keywords for every dataset. I used this to tokenize all the keywords. Then I built a CBOW-model ("continuous bag of words"-model, specs see below). I then trained the tokenized words in several epochs (again, see more below). Lastly, I visualized the results and played around a bit with DBSCAN to try to make sense of it (spoiler: I found very little).

From the website (https://www.govdata.de/web/guest/hilfe): "GovData, the data portal for Germany, offers uniform, centralized access to administrative data from the federal, state and local governments as well as data from public service companies, universities, researchers, research institutions and research funding bodies. The aim is to make it easier to find and use this data in one place."

### Some information and results
- Data format of the GovData metadata: https://www.dcat-ap.de/def/dcatde/2.0/spec/
- Dataset with the most keywords: https://www.govdata.de/web/guest/suchen/-/details/auslander-kreise-stichtag-geschlecht with 720 keywords
- Funniest keyword I could find: 17-abs.3-aufenthg-ae-arb.platzsuche-n.-betr.-ba (yep that one keyword)
- Longest keyword: "richtlinie_des_ministeriums_für_infrastruktur_und_landwirtschaft_über_die_gewährung_von_zuwendungen_für_die_förderung_der_integrierten_ländlichen_entwicklung_(ile)_und_leader_vom_13._nov._2007_zuletzt_geändert_am_23._dezember_2013;_teil_f-natürliches_erbe" (this one is cool too)
- Structure of the CBOW-model: keras.models.Sequential(), one Embedding layer, one Lambda layer (lambda x: tf.reduce_mean(x, axis=1), output_shape=(embedding_size,)), one Dense layer (activation='softmax'), then compiled (loss='categorical_crossentropy', optimizer='rmsprop'), embedding size was 2
- Training: 10 epochs, each on the complete dataset and taking 2.5 hours

### Structure of this repository
- Braune_Hausarbeit.pdf: My term paper, it's in German
- Program01.py: The program I wrote
- Program01_corpus.csv: One row equals one dataset. For each, all of the keywords are listed (whitespaces inside keywords are changed to underscores)
- Program01_embeddings.csv: 10-dimensional embeddings
- Program01_identifiers.csv: One row equals one dataset. For each, their respective identifier is listed (the order is the same as in the corpus)
- Program01_keyword_counts.csv: For each keyword-ID, their total number of appearances over the data is listed
- Program01_reduced_embeddings.csv: 2-dimensional embeddings
- Program01_titles.csv: One row equals one dataset. For each, their respective title is listed (the order is the same as in the corpus)
- Program01_word_index.csv: For each keyword, their respective ID (as assigned through tokenization) is listed
- Missing here is "Program01_keywords.csv": The original query result. One row includes one dataset title, their identifier and ONE of their keywords

License: CC-0, so feel free to use it. It would be awesome to see examples if it does get used further

### Learnings
- I really want to know where these keywords come from. Does some poor soul sit there and type them in? Are they auto-generated? Some weird mix? Do different institutions handle this differently?
- Not really a learning, but I also want to know if it is even possible to train these keywords, and if yes what I did wrong
- No proper versioning. E.g., In the paper I wrote I only did embeddings of size 2, but here there is a file with embedding size 10...

## Background
I wrote this script for my programming course on machine learning at the "Digital Humanities" Master of Science at Trier University in the summer semester 2023. In the seminar, we learned several machine learning algorithms (k-Means, DBScan, k-Nearest Neighbour, Neural Networks) and how to implement them by hand in Python (well, except for the neural networks. Had to use a library with that one). That is to say, I'm not an expert. But I still had fun! I might've had too much fun looking at what people used as "keywords" in those datasets...

I chose this topic, because I'm interested in Open Data and thought it was an interesting project. I still think so, although I currently don't have the time it needs for the results to be properly usable.

## Disclaimer
I did have prior experience wiht Python, SQL, and some of the libraries before writing this. However, I've only done one other scraping program in Python before this. So please expect there to be some errors, anti-patterns, and most probably awkward code. I would be grateful if you pointed those out to me if you found more, so I can add them to the list below. You can mail me at info@wandvieh.de.

### What could have been done better
- Downloading the keywords: The way I download, is that I download all dcat:Dataset (s) that have a dct:title, at least one dcat:keyword AND an dct:identifier. The problem with this: The identifier is not obligatory. So all datasetes without an identifier are left behind
- Tokenization: There might be an error with the tokenization. I get keywords like "\_gemeinden", but I'm not sure those should be in there. It might have to do with the way I handled the keywords: Any space characters, I replaced with "\_" so they wouldn't be torn apart
- The CBOW model: I used a tutorial for the model, but can't say that I've fully understood it. Meaning I also can't say in detail whether this could've been done better
- Again the CBOW model: Maybe this wasn't the best choice. I want every keyword to be trained with every other keyword in its dataset. Either use a different model for that or increase window size to 720 (the maximum number of keywords in a dataset). Maybe should've used Skipgram?
- Dimensionality: I only did an embedding size of 2 (= 2-dimensional vector), which I'm strongly suspecting to have been a mistake. I haven't yet tried more dimensions, since training took quite a while and I've not been in the mood to do it again yet
- DBSCAN / Epsilon: I might have not chosen this properly. I used an internet tutorial for how to choose this, but either there was some mistake or my data really is just that wonky
