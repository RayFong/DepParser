# DepParser
The project is about dependency parsing with empty categories

Data format:
One word per line, sentence is separated by empty line.
For each line, we have: word, POS tag, head word index, dependency type (separated by single tab)
Word index starts from 0 for each sentence, and -1 represents root.
Empty nodes are represented by *XX*, for example *PRO*.
