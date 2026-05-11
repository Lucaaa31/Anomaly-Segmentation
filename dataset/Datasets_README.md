## 1. Cityscapes
**Focus:** Segmentazione semantica urbana.

**Contenuto:** Immagini ad alta risoluzione di scenari stradali in 50 città diverse.

**Caratteristiche:** Annotazioni a livello di pixel per 30 classi (pedoni, veicoli, segnaletica, ecc.). 

## 2. FS_LostFound_full (Fishyscapes Lost & Found)
**Focus:** Rilevamento di piccoli ostacoli sulla carreggiata.

**Contenuto:** Estratto dal dataset Lost and Found, rielaborato per il benchmark Fishyscapes.

**Caratteristiche:** Include oggetti reali (giocattoli, scatole, detriti) lasciati sulla strada che non rientrano nelle classi standard di Cityscapes.

## 3. fs_static (Fishyscapes Static)
**Focus:** Rilevamento di anomalie sintetiche.

**Contenuto:** Immagini di Cityscapes con l'inserimento digitale di oggetti estranei (es. animali, oggetti domestici).

**Caratteristiche:** Utilizzato per testare la capacità di un modello di segnalare come "incerto" o "anomalo" qualcosa che non ha mai visto durante il training.

## 4. RoadAnomaly / RoadAnomaly21
**Focus:** Anomalie del mondo reale e scenari atipici.

**Contenuto:** Immagini pescate dal web che contengono situazioni insolite (es. un aereo in autostrada, animali selvatici, detriti giganti).

**Caratteristiche**: Serve a valutare la robustezza del software di fronte a eventi "Black Swan" (estremamente rari).

## 5. RoadObstacle21
**Focus:** Ostacoli generici sulla traiettoria.

**Contenuto:** Simile a RoadAnomaly, ma focalizzato specificamente su oggetti che bloccano il percorso del veicolo.

**Caratteristiche:** Fa parte dei dataset curati per il benchmark SegmentMeIfYouCan, mirato a distinguere tra ciò che è strada percorribile e ciò che è un ostacolo ignoto.