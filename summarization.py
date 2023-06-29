import yazlab2_3
import json
import string
from nltk.stem.porter import *

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertModel
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import BertTokenizer

# import torch, transformers, tokenizers
# torch.__version__, transformers.__version__, tokenizers.__version__

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

data_file = 'test.json'

# Verilerin satır satır okunması
def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line

metadata = get_metadata()
for paper in metadata:
    for k, v in json.loads(paper).items():
        print(f'{k}: {v} \n')
    break
titles = []
abstracts = []
texts = []
'''
metadata = get_metadata()
for paper in tqdm(metadata):
    paper_dict = json.loads(paper)
    titles.append(paper_dict.get('title'))
    abstracts.append(paper_dict.get('tgt').replace("\n",""))
    texts.append(paper_dict.get('src').replace("\n",""))

if len(titles) != len(abstracts):
  raise Exception("Başlık ve Özet uzunlukları eşit değil")
'''
texts.append("We describe a convolutional neural network that learns feature representations for short textual posts using hashtags as a supervised signal. The proposed approach is trained on up to 5.5 billion words predicting 100,000 possible hashtags. As well as strong performance on the hashtag prediction task itself, we show that its learned representation of text (ignoring the hashtag labels) is useful for other tasks as well. To that end, we present results on a document recommendation task, where it also outperforms a number of baselines.")
titles.append("#TagSpace: Semantic Embeddings from Hashtags")
abstracts.append("A convolutional neural network model for predicting hashtags was proposed in REF .")

papers = pd.DataFrame({
    'title': titles,
    'input_text': texts,
    'output_text': abstracts
})



# Null sayısınının yazdırılması 
print(papers.isnull().sum())

# Null değerlerin çıkarılması
papers = papers.dropna()

del titles, abstracts, texts

sentences_row = []

# Iterate over each element in the column
for element in papers['input_text']:
    elements = element.split(". ")
    sentences_row.append(elements)

# Remove empty strings
for sentences in sentences_row[:]:
    for sentence in sentences[:]:
        if(len(sentence) == 0):
            sentences.remove('')

# Her row başında 1 tane boşluk karakteri var

sentences2 = []
sentences_row_words = []

# Iterate over each element in the column
for sentences in sentences_row[:]:
    sentences2 = []
    for sentence in sentences[:]:
        words = word_tokenize(sentence)
        sentences2.append(words)
    sentences_row_words.append(sentences2)

sentences_row_nopunctuation = []

for sentences in sentences_row[:]:
    temp_sentences = []
    for sentence in sentences[:]:
        cleared = ""
        for char in sentence[:]:
            if char not in string.punctuation:
                cleared += char
        temp_sentences.append(cleared)
    sentences_row_nopunctuation.append(temp_sentences)

    sentences_row_words_nopunctuation = []
temp_sentences = []
temp_words = []

# Iterate over each element in the column
for sentences in sentences_row_words[:]:
    temp_sentences = []
    for sentence in sentences[:]:
        temp_words = []
        for word in sentence[:]:
            cleared = ""
            for char in word[:]:
                if char not in string.punctuation:
                    cleared += char
            temp_words.append(cleared)
        temp_sentences.append(temp_words)
    sentences_row_words_nopunctuation.append(temp_sentences)

# Remove empty strings
for sentences in sentences_row_words_nopunctuation[:]:
    for sentence in sentences[:]:
        for word in sentence[:]:
            if(len(word) == 0):
                sentence.remove('')

for sentences in sentences_row_words_nopunctuation[:]:
    for sentence in sentences[:]:
        if(len(sentence) == 0):
            sentences.remove(sentence)

title_row = []

# Iterate over each element in the column
for element in papers['title']:
    title_row.append(element)

output_row = []

# Iterate over each element in the column
for element in papers['output_text']:
    output_row.append(element)


# Her sentences başında 1 tane boşluk karakteri var

digit_counts = 0
digit_parameter = []
digit_parameters = []

# Iterate over each column in the dataframe
for sentences in sentences_row[:]:
    digit_parameter = []
    # Iterate over each element in the column
    for sentence in sentences[:]:
        sentence_length = len(sentence)
        digit_counts = 0
        digit_score = 0
        for char in sentence:
            # Calculate the character count and append it to the character_counts dataframe
            if(char.isdigit()):
                digit_counts = digit_counts + 1
        digit_score = digit_counts / sentence_length
        digit_parameter.append(digit_score)
    digit_parameters.append(digit_parameter)

def count_nouns(text):
    tokens = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(tokens)
    proper_noun_count = 0
    non_proper_noun_count = 0

    for word, tag in tagged_words:
        if tag == 'NNP' or tag == 'NNPS':
            proper_noun_count += 1

    return proper_noun_count

proper_parameter = []
proper_parameters = []

for sentences in sentences_row_words[:]:
    proper_parameter = []
    for sentence in sentences[:]:
        sentence_length = len(sentence)
        proper_counts = count_nouns(' '.join(sentence))
        proper_score = proper_counts / sentence_length
        proper_parameter.append(proper_score)
    proper_parameters.append(proper_parameter)

# Kelimelerin geçtiği sayıları saklamak için boş bir liste oluşturun
title_input_text_parameters = []
title_input_text_parameter = []
i = 0

for sentences in sentences_row_words_nopunctuation[:]:
    title_input_text_parameter = []
    for sentence in sentences[:]:
        sentence_length = len(sentence)
        # Her bir kelimeyi title ile karşılaştırın
        title_counts = sum(1 for word in sentence if word in title_row[i])
        title_score = title_counts / sentence_length
        title_input_text_parameter.append(title_score)
    i = i+1
    title_input_text_parameters.append(title_input_text_parameter)


#stopset = set(stopwords.words('english'))
#vectorizer = TfidfVectorizer(use_idf = True , strip_accents='ascii' , stop_words = stopset, #tokenizer=True
                            #)

predicted_abstract_list = []
true_abstract_list = []

for i in range(1):
    #XX = vectorizer.fit_transform(sentences_row_nopunctuation[i])
    #vectorizer.get_feature_names()
    
    #####################################################################################################
    
    # Öznitelik isimlerini alın
    #feature_names = np.array(vectorizer.get_feature_names())

    # XX matrisini sıkıştırarak sadece skorları içeren bir numpy dizisine dönüştürün
    scores = []  # Skorları tutmak için boş bir liste oluşturun
    '''
    
    for row in range(XX.shape[0]):
        for col in XX[row].nonzero()[1]:
            word = feature_names[col]
            score = XX[row, col]
            scores.append((word, score))  # Cümle skorlarını ana listeye ekleyin

    # Skorları yazdırma yerine kaydetmek isterseniz, farklı bir değişkene atayabilirsiniz.
    # Örneğin:
    saved_scores = scores

    # Kelimeleri skorlarına göre büyükten küçüğe sıralayın
    sorted_scores = sorted(saved_scores, key=lambda x: x[1], reverse=True)  # İndeksleri sıralama sırasını tersine çevirin

    top_10_percent = int(len(sorted_scores) * 0.1)  # Toplam skor sayısının %10'u
    top_10_scores = sorted_scores[:top_10_percent]  # İlk %10'luk skorları al

    top_10_words = [score[0] for score in top_10_scores]  # Skorların kelimelerini al ve bir listeye kaydet

    # Elde edilen %10 kelimeleri yazdırın
#     print(top_10_words)
    
    #####################################################################################################
    
    theme_words = ""
    
    theme_words = " ".join(top_10_words)

    theme_paremeters = []

    for sentence in sentences_row_nopunctuation[i]:
        sentence_length = len(sentence)
        theme_counts = sum(1 for word in sentence if word in theme_words)
        theme_score = theme_counts / sentence_length
        theme_paremeters.append(theme_score)
#         print(" {}, \n{}, \n{}".format(sentence, theme_words, theme_counts))
        
    #####################################################################################################
    '''
    #Initialize our model and tokenizer:
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    ###Tokenize the sentences like before:
    sent = sentences_row_nopunctuation[i]

    # initialize dictionary: stores tokenized sentences
    token = {'input_ids': [], 'attention_mask': []}
    for sentence in sent:
        # encode each sentence, append to dictionary
        new_token = tokenizer.encode_plus(sentence, max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        token['input_ids'].append(new_token['input_ids'][0])
        token['attention_mask'].append(new_token['attention_mask'][0])
    # reformat list of tensors to single tensor
    token['input_ids'] = torch.stack(token['input_ids'])
    token['attention_mask'] = torch.stack(token['attention_mask'])
    #Process tokens through model:
    output = model(**token)
    output.keys()
    
    #####################################################################################################
    
    #The dense vector representations of text are contained within the outputs 'last_hidden_state' tensor
    embeddings = output.last_hidden_state
    embeddings
    
    #####################################################################################################
    
    # To perform this operation, we first resize our attention_mask tensor:
    att_mask = token['attention_mask']
    att_mask.shape
    
    #####################################################################################################

    mask = att_mask.unsqueeze(-1).expand(embeddings.size()).float()
    mask.shape
    
    #####################################################################################################
    
    mask_embeddings = embeddings * mask
    mask_embeddings.shape
    
    #####################################################################################################

    #Then we sum the remained of the embeddings along axis 1:
    summed = torch.sum(mask_embeddings, 1)
    summed.shape
    
    #####################################################################################################

    #Then sum the number of values that must be given attention in each position of the tensor:
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    summed_mask.shape
    
    #####################################################################################################

    mean_pooled = summed / summed_mask
    mean_pooled
    
    #####################################################################################################

    #Let's calculate cosine similarity for sentence 0:
    # convert from PyTorch tensor to numpy array
    mean_pooled_list = mean_pooled.detach().numpy()
    # calculate
    # cosine_similarity(
    #     [mean_pooled[0]],
    #     mean_pooled[1:]
    # )

    similarity = []
    sim = []

    for j in range(len(mean_pooled_list)):
        result = cosine_similarity([mean_pooled_list[j]],mean_pooled_list[:])
        similarity.append(result.tolist())
        sim.append(result.tolist())

    print(sim)
    print("------")
    print(similarity)
    

    for j in range(len(mean_pooled_list)):
        similarity[j][0].pop(j)
#     print(similarity)
    #         print(f"Cümle {i} ile Cümle {j} arasındaki benzerlik: {similarity}")
    
    #####################################################################################################

    sentence_similarity_threshold = 0.7
    sentence_score_threshold = 0.5
    
    #####################################################################################################

    sentence_similarity_threshold_counter = []

    for j in range(len(mean_pooled_list)):
        counter = 0
        for k in range(len(mean_pooled_list)-1):
            if similarity[j][0][k] > sentence_similarity_threshold:
                counter += 1
        sentence_similarity_threshold_counter.append(counter)
    sentence_similarity_threshold_counter
    
    #####################################################################################################

    sentence_similarity_threshold_parameters = []

    for j in range(len(mean_pooled_list)):
        sentence_similarity_threshold_parameters.append(sentence_similarity_threshold_counter[j]/len(mean_pooled_list))

    sentence_similarity_threshold_parameters
    
    #####################################################################################################

    proper_rate = 0.5
    digit_rate = 0.25
    similarity_rate = 0.5
    title_rate = 0.75
    theme_rate = 0.5

    final_scores = []
    
    for j in range(len(sentence_similarity_threshold_parameters)):
        print("aa")
        sc = 0
        sc += proper_rate*proper_parameters[i][j]
        sc += digit_rate*digit_parameters[i][j]
        sc += similarity_rate*sentence_similarity_threshold_parameters[j]
        sc += title_rate*title_input_text_parameters[i][j]
        #sc += theme_rate*theme_paremeters[j]
        final_scores.append(sc)

    final_scores
    finalsc = final_scores
    #####################################################################################################

    final_scores = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
    final_scores
    
    #####################################################################################################

    predicted_abstract = ""
    print("asa",sentences_row)
    for j in range(int(format(len(final_scores)*3/10, ".0f"))):
        if j == int(format(len(final_scores)*3/10, ".0f"))-1:
            
            predicted_abstract += sentences_row[i][final_scores[j][0]] + "."
        else:
            predicted_abstract += sentences_row[i][final_scores[j][0]] + ". "

    true_abstract = output_row[i]
    print(f"Predicted Abstract : {predicted_abstract}\n\nTrue Abstract : {true_abstract}\n\n\n")

    predicted_abstract_list.append(predicted_abstract)
    true_abstract_list.append(true_abstract)

    import evaluate

#use_aggregator : if False;Her prediction için ayrı ayrı skorlar verilir. if True;Ortalaması alınır.
#use_stemmer : if True;Kelimenin köklerine indirgeyerek, kök halleri kullanılarak aynı olarak kabul edileceği anlamına gelir.
#if False; Metindeki her kelimeyi tam olarak dikkate alır.

rouge = evaluate.load('rouge')
results = rouge.compute(predictions=predicted_abstract_list,
                        references=true_abstract_list,
                        use_aggregator=True,
                        use_stemmer=False
                       )
print(results)

#ROUGE1, yalnızca özet ve referans özeti arasındaki tam eşleşen 1-gram kelimeleri kullanırken
#ROUGE2, 2-gram kelimeleri kullanır.
#ROUGEL, özet ve referans özeti arasındaki en uzun ortak alt dizgiyi bulmak için Longest Common Subsequence (LCS) kullanırken
#ROUGE Lsum, ROUGE L için farklı L uzunlukları için sonuçları birleştirir.

#ROUGE1, ROUGE2 ve ROUGE L ölçüleri, özetlemenin farklı yönlerini değerlendirir
#Bir özetlemenin kalitesini ölçmek için birbirleriyle birlikte kullanılır
#ROUGE Lsum, farklı L uzunlukları için sonuçları birleştirerek, özetlemenin genel kalitesini ölçmek için kullanılabilir.
import evaluate

#max_order : Buradaki girilecek olan sayı kaçsa n-gram'larına bakarak bir başarı üretir.
#smooth : Yüksek smooth değerleri, uzun çevirilerin performansını artırırken kısa çevirilerin performansını düşürebilir.
#Düşük smooth değerleri ise kısa çevirilerin performansını artırırken uzun çevirilerin performansını düşürebilir.
#0 ya da 1 tercih edilir.
bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predicted_abstract_list,
                       references=true_abstract_list,
                       max_order=4,
                       smooth=False
                      )
print(results)

#bleu : BLEU puanı, 0 ile 1 arasında bir değerdir.
#precisions : "precisions" değeri, farklı n-gram düzeylerindeki precision değerlerini içeren bir dizi veya sözlüktür.
#brevity_penalty: Brevity penalty, çeviri cümlesinin referans cümleye göre ne kadar kısa olduğunu dikkate alır
#Bu, çeviri cümlesi uzunluğunun referans cümlesine göre ne kadar farklı olduğunu ölçer ve bir ceza uygular.
#length_ratio : Çeviri cümlesinin uzunluğu ile referans cümlesinin uzunluğu arasındaki oranı gösterir. Daha kısa bir çeviri cümlesi için bu oran daha azdır.
#translation_length : Çeviri cümlesinin uzunluğunu temsil eder. Bu, çeviri cümlesindeki toplam kelime veya karakter sayısını ifade edebilir.
#reference_length : Referans cümlesinin uzunluğunu temsil eder. Bu, referans cümlesindeki toplam kelime veya karakter sayısını ifade edebilir.

def draw_graph(self):
    print(sentences_row)
    graph = yazlab2_3.Graph(sentences_row[0],finalsc,sentence_similarity_threshold_parameters,sim)
    graph.show_graph()