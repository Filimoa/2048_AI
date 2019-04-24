



import re


string = "sergey1345-"

regex = re.compile('[^a-zA-Z]')

newString = regex.sub('', string)

print newString

