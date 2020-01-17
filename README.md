# BinaryStereoMatching
## Beschreibung:
Teilweise Implementierung eines auf dem BRIEF-Deskriptor basierenden Stereometrieverfahrens.
## Quellen:
### Algorithmus:
K. Zhang et al., Binary Stereo Matching, ICPR 2012
### Rektitfizierte Stereobilder:
http://vision.middlebury.edu/stereo/

## Abhängikeit:
OpenCV

## Verwendung:
BinaryStereoMatching.exe <eingabebild_links> <eingabebild_rechts> <n> <threads>

## Argumente:
eingabeBild_links, eingabeBild_rechts:
	Pfade zu den Eingabebildern
n:
	Länge des Deskripors in Bits. Default: 4096
threads:
	Anzahl der zu verwendenden Threads. Default: Maximale Zahl paralleler Threads.
	
## Output:
	result.png
		Bild aus Disparitäten zwischen 0 und 255
	result_equalized.png
		normalisiertes Ergebnis, mittels opencvfunktion equalizeHist()
## Beispiel:
BinaryStereoMatching.exe .\inputs\im0.png .\inputs\im1.png 4096 1
BinaryStereoMatching.exe .\inputs\im0.png .\inputs\im1.png 4096
BinaryStereoMatching.exe .\inputs\im0.png .\inputs\im1.png