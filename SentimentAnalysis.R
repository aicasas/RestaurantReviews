
# Sentiment Analysis: Bag of Words. Natural Language Processing (NLP)
#ACCURACIES:
# SVM: 79.5%
# k-NN:69.2%
# Logistic Regression:55%


#Reading Text File
data <- read.delim(file = 'Restaurant_Reviews.tsv', quote="", stringsAsFactors = FALSE)
#sep = '\t', header = TRUE)

#Cleaning Text
which(is.na(data))
is.na(data)
install.packages('tm')
install.packages('SnowballC')

library(tm)
library(SnowballC)

#corpus is the bag of words (working on Review Column)
corpus= VCorpus(VectorSource(data$Review)) #Use column review to create corpus
corpus=tm_map(corpus, content_transformer(tolower)) #make everything lowercase
corpus = tm_map(corpus, removeNumbers) #numbers 
corpus=tm_map(corpus, removePunctuation) #punctuation (..., !, ?)
corpus=tm_map(corpus, removeWords, stopwords()) #stopwords (the, a, who...)
corpus= tm_map(corpus, stemDocument) #stem of word (aparent is stem of apparently)
corpus=tm_map(corpus, stripWhitespace) #whitespaces
corpus[[1]][[1]]

#creating the bag of words model
dtm= DocumentTermMatrix(corpus)
dtm =removeSparseTerms(dtm, 0.999 ) #reduction of sparsity (if word is not found often)
dataset =as.data.frame(as.matrix(dtm)) #concerting from list to matrix / dataframe
dataset$Liked= data$Liked #insert liked colum from the original data into dataset

#Encoding the liked variable
dataset$Liked= factor(dataset$Liked, levels =c(0,1))

library(caTools) #activating package 
set.seed(123)
#Trying to predict whether they liked it 
split=sample.split(dataset$Liked, SplitRatio=0.8)
#splitting ratio can be anything. If it is higher than 95% then you are overfitting. 
#It is too familiar with the data that it can't understand the. The ratio where you get the most results should be picked

training_set=subset(dataset, split==TRUE)
test_set=subset(dataset, split==FALSE)


#Applying k-NN
#k-Nearest Neighbors #k can be any number
#install.packages('class')
library(class)
y_pred = knn(train = training_set[,-692],
             test = test_set[,-692],
             k=5,
             cl = training_set[,692],prob=TRUE)
#You train  on the training set, test on the test set, 5 nearest neighbors,
#cl is the response variable - the one you are predicting

#Confusion Matrix
con_matrix <- table(test_set[,692], y_pred)
accuracy=(75+57)/200
print(accuracy) #69.2%


#Apply Logistic Regression
classifier = glm(formula = Liked~.,family = binomial, data = training_set)
pro_pred = predict(classifier,type='response',test_set[,-692])   
y_pred_2 = ifelse(pro_pred>0.5,1,0)
#Confusion Matrix
con_matrix_2 = table(y_pred_2, test_set[,692])
accuracy_logistic= (49+61)/200
print(accuracy_logistic) #55%. 

#Applying SVM
library(e1071)
classifier=svm(formula = Liked~., data=training_set, 
               type = 'C-classification', kernel= 'linear')
#kernal = linear is enforcing a linear classification. This is not always true nor most popular
#Kernal is an instance 
#C-classification is based on on maximizing the distance 
# it will predict a 1 or 0 rather than a number of 1s of 0s 
y_pred=predict(classifier, newdata = test_set[,-692])
con_matrix_3 <- table(test_set[,692], y_pred)
accuracy3=(78+81)/200
print(accuracy3) #79.5% Accuracy. Seems to be the best.
Results <- as.matrix(con_matrix)

#Creating Cloud
install.packages("wordcloud")
install.packages("RColorBrewer")
library(wordcloud)
library(RColorBrewer)

DTM =TermDocumentMatrix(corpus)
matrixx <- as.matrix(DTM)
frq <- sort(rowSums(matrixx), decreasing= TRUE)
dat <- data.frame(word=names(frq), freq=frq)
head(dat)

#The Word Cloud
set.seed(101)
wordcloud(words=dat$word, freq=dat$freq)
#word cloud in ordered manner
wordcloud(words=dat$word, freq=dat$freq, random.order=FALSE) 
#highest most freq in center, less freq aroud the edges 
#Word Cloud with Colors
wordcloud(words=dat$word, freq=dat$freq, colors= brewer.pal(8, "Dark2"), random.order=FALSE)



