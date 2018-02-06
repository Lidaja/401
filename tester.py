from a1_preproc import *


def run_tests(s,o,f,n):
	for i in range(len(s)):
		assert f(s[i]) == o[i], "\nStep: "+str(n)+" - The function returned:\n-----------------\n\""+f(s[i])+"\"\n------------\n but expected:\n------------------\n\""+o[i]+"\"\n----------------"
	

if __name__ == '__main__':


	#Step 1
	sentences = ["This is a\n test sentence."]
	output = ["This is a test sentence."]
	run_tests(sentences,output,remove_newline,1)

	#Step 2
	sentences = ["Go to www.lijax.com for cool stuff","My favourite httplace to go is http://youtube.com"]
	output = ["Go to for cool stuff", "My favourite to go is"]
	run_tests(sentences,output,remove_urls,2)

	#Step 3
	sentences = ["This is a wicked &amp; sweet test"]
	output = ["This is a wicked & sweet test"]
	run_tests(sentences,output,replace_html,3)
	
	#Step 4
	sentences = ["leddit.I","UU=======)**:-D**","How do you do?","Don't go in there!", "How!?!? is that god-damn possible.", "Teach me to use A.E","and doggone it, people like me....\"You little liar!"]
	output = ["leddit . I","UU =======)**:- D **","How do you do ?", "Don't go in there !", "How !?!? is that god - damn possible .", "Teach me to use A.E","and doggone it , people like me ....\" You little liar !"]
	run_tests(sentences,output,split_punctuation,4)

	#Step 5
	sentences = ["He couldn't do it", "The dog's dogs' toys are what we're waiting for"]
	output = ["He could n't do it", "The dog 's dog s' toys are what we 're waiting for"]
	run_tests(sentences,output,split_clitics,5)

	#Step 6
	sentences = ["I also walked to the store"]
	output = ["I/PRP also/RB walked/VBD to/IN the/DT store/NN"]
	run_tests(sentences, output,spacy_tag,6)

	#Step 7
	sentences = ["I/PRP also/RB walked/VBD to/IN the/DT store/NN", "High noon"]
	output = ["walked/VBD store/NN", "noon"]
	run_tests(sentences,output,remove_stopwords,7)

	#Step 8
	sentences = ["walked/VBD store/NN"]
	output = ["walk/VBD store/NN"]
	run_tests(sentences,output,lemmatize,8)

	#Step 9
	sentences = ["walk/VBD St/NNP ./. bernard/NN store/NN ./. rude/JJ"]
	output = ["walk/VBD St/NNP ./. bernard/NN store/NN ./. \n rude/JJ"]
	run_tests(sentences,output,add_newline,9)

	#Step 10
	sentences = ["Walk/VBD St/NNP ./. Bernard/NN store/NN ./. \n rude/JJ"]
	output = ["walk/VBD st/NNP ./. bernard/NN store/NN ./. \n rude/JJ"]
	run_tests(sentences,output,make_lowercase,10)

	print("ALL TESTS PASS!")
	

