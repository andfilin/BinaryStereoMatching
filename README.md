# BinaryStereoMatching
## Beschreibung:
Teilweise Implementierung eines auf dem BRIEF-Deskriptor basierenden Stereometrieverfahrens.
## Quellen:
### Algorithmus:
K. Zhang et al., Binary Stereo Matching, ICPR 2012
### Rektitfizierte Stereobilder:
http://vision.middlebury.edu/stereo/

## Abh�ngikeit:
OpenCV

## Verwendung:
BinaryStereoMatching.exe *eingabebild_links* *eingabebild_rechts*  
Alle Parameter k�nnen in der Datei definitions.h angepasst werden.
	
## Output:
* result.png  
Bild aus Disparit�ten zwischen 0 und 255
* result_equalized.png  
normalisiertes Ergebnis, mittels opencvfunktion equalizeHist()
