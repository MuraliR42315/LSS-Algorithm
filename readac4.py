import pysrt

import nltk
#from pathlib import Path
import re
import csv
from nltk import parse,grammar,data
import os
import glob
from nltk.corpus import stopwords

ttrl=[]
Wordamgl=[]
sentambl=[]
totreadl=[]
ftime=[]
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import treebank

f= open('rdmvs.csv','w')
writer= csv.writer(f)
filen=[]
grammar = data.load('unigrammar.fcfg')
uparser = parse.FeatureChartParser(grammar)
'''opening a srt file'''
#p=Path("/home/muralidhar/Downloads/captionfiles/")
#time=0

for filename in glob.glob('/home/muralidhar/test movies/*.srt'):
	try:
		print (filename)
		subs = pysrt.open(filename)
		time=subs[0].start
		sentences=""
		filen.append(filename)
		''' Creating a paragraph with the subtitles'''
		for i in range(len(subs)):		
			time=time+subs[i].end-subs[i].start
			sentences = sentences+' '+subs[i].text
	
		time=time-subs[0].start	
		time= time.hours*60*60+time.minutes*60+time.seconds
		time=time/len(subs)			
		#print (time/len(subs))
		ftime.append(time)
		#print sentences
		#filen.append(filename)
		'''Removing the stopWords'''
		stopword=stopwords.words('english')
		words = word_tokenize(sentences)
	
		wordsFiltered = []

		for w in words:
			if w not in stopword:
				b=".'-,/"
				for char in b:
					w=w.replace(char,"")
				if w.isalpha():	
					wordsFiltered.append(w)
	

		wordlen= len(wordsFiltered)

		worddis=[]

		for w in wordsFiltered:
			if w not in worddis:
				worddis.append(w)


		ttr= float(len(worddis))/len(wordsFiltered)
		ttr=round(ttr,4)	
#	print ttr

		ttrl.append(ttr)

		sum=0
		for w in worddis:
			if w.isalpha():
				c=0
				synsets = wn.synsets(w)
				if len(synsets)>0:
					for l in synsets:
						if len(l.hyponyms())>0:
							c=c+1
			if c>=2:
				sum=sum+1
		#print sum
		#print len(worddis)
#sentences = stopwords.clean(sentences.lower().split(),"en")
		Wordamg= float(sum)/len(worddis)
#	print Wordamg
		Wordamg=round(Wordamg,4)
		Wordamgl.append(Wordamg)

		sents=sent_tokenize(sentences)
#print Wordamgl
#print len(sent)

		wordtags = nltk.ConditionalFreqDist((w.lower(), t)
		for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))
		sum=0
		amb=0
	
		for sent in sents:
			word=word_tokenize(sent)
			c=0
			ambi=0
#	print sent
			for i in word:
				if i not in stopword:
					if i.isalpha():
						a=list(wordtags[i])
						if 'VERB' in a:
							c=c+1	
				if c>1:
					ambi=ambi+1			
			if ambi>1:
				amb=amb+1
		sentamb= float(amb)/len(sents)
#		print sentamb
		sentamb=round(sentamb,4)
		sentambl.append(sentamb)
#		print sentamb
		totread = (ttr+Wordamg+sentamb)/time
		#print (totread)
#		print totread
		totread=1-totread
		print(totread)
		totread=round(totread,4)
		totreadl.append(totread)
		#print (filename)
	except (UnicodeDecodeError,IndexError):
		pass

for j in range(len(ttrl)):
#	f.writerows(str([filen[i]]+[ttrl[i]]+[sentambl[i]]+[Wordamgl[i]]+[totreadl[i]])
	f.write(str(filen[j])+","+str(ttrl[j])+","+str(sentambl[j])+","+str(Wordamgl[j])+","+str(ftime[j])+","+str(totreadl[j])+"\n")

#	print (filen[i])
#	print ()
#	print ()
#	print()
