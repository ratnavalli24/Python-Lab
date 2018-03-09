
with open('input.txt' , 'r') as file:
    lines = file.readlines()
print("lines",lines)
first= ''

for m in lines:
    first= first + m
print(first)

first_word = word_tokenize(first)
first_sent = sent_tokenize(first)


lm = WordNetLemmatizer()
first_lma = []
for word in first_word:
    first_lma = lm.lemmatize(word.lower())
    first_lma.append(first_lma)

print("lemmetaizion")
print(first_lma)
fr_pos = pos_tag(first_lma)

print("BIGRAM")

a = 2
gm=[]
bigrams = ngrams(first_lma, a)
for grams in bigrams:
    gm.append(grams)
print(gm)
str1 = " ".join(str(x) for x,y in fr_pos)
str1_word = word_tokenize(str1)
dst1 = nltk.FreqDist(gm)
Tp5 = dst1.most_common()
top_five = dst1.most_common(5)

Tp=sorted(Tp5, key=itemgetter(0))
print(Tp)
print(top_five)
snt = sent_tokenize(first)
rep_sent1 = []


for sent in snt:
    for word,words in gm:
        for ((c,m), l) in top_five:
            if (word,words == c,m):
                rep_sent1.append(sent)
print ("\n Sentences with top five Bigrams")
print(max(rep_sent1,key=len))
