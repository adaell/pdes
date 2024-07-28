
docs:
	xelatex readme.tex

clean:
	rm -f readme.log
	rm -f readme.aux
	rm -f readme.pdf
	rm -f readme.blg
	rm -f readme.dvi
	rm -f readme.bcf
	rm -f readme.run.xml
	rm -f images/*
	rm -f animation.gif
