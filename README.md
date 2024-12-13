Python-filer brukt i bachelor: "Automatisk maritim overvåkning - kunstige nevrale nettverk anvendt for klassifisering av hydroakustiske lydsignaler fra fartøy" av Marius Polsrød og Sivert Karlsen

Fremgangsmåte:
1. Last ned datasett fra link:
-  DeepShip: https://github.com/irfankamboh/DeepShip
-  LoVe: https://www.jottacloud.com/s/303431012aa75bc484bac9a1dfced29de86/thumbs
2. Del opp i 3 sekunders segmenter: Preprocess/trim.py
3. Fordel på subsett: Preprocess/split_dataset.py
4. Bruk ønsket augmenteringsmetode (alternativt)
- konstant frekvensforksyvning: Preprocess/wav_ps.py
- tidsavhengig frekvensforskyvning: Preprocess/wav_ips.py
5. Konverter fra wav. til .json med cqt-transform: Preprocess/was_to_cqt.py
- RNN: Train/RNN.py
- CNN: Train/RNN.py

OBS! På grunn av størrelsen på datasettene kreves det veldig mye datakraft for å kjøre opptreningsprogrammene. En maskinklynge ble derfor brukt til dette i bacheloroppgaven
Kan søke tilgang med link: https://www.ex3.simula.no/access
For testing og kjøring av maskinlæringsmodellene kan "...mini.json" brukes



 
