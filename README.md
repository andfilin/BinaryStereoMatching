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
BinaryStereoMatching.exe *eingabebild_links* *eingabebild_rechts*  
Alle Parameter können in der Datei definitions.h angepasst werden.
	
## Output:
* result.png  
Bild aus Disparitäten zwischen 0 und 255
* result_equalized.png  
normalisiertes Ergebnis, mittels opencvfunktion equalizeHist()
