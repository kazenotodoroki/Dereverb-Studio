# Dereverb Studio

Retirer une réverbération **connue** de pistes audio, à partir de son IR — ou d'une capture faite avec le signal de mesure intégré (sweeps sinus exponentiels, comme Altiverb).

## Installation

Python 3.10+ :

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate   —   macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Pour un GPU NVIDIA, installez PyTorch avec CUDA depuis pytorch.org avant les autres paquets : l'entraînement neuronal sera 20 à 50× plus rapide. Sur Mac Apple Silicon, le GPU (MPS) est utilisé automatiquement.

## Quelle méthode pour quel cas ?

Une règle de physique décide de tout : **une IR n'est exacte que pour la position de source où elle a été mesurée.** Dans l'aigu, déplacer la source de quelques centimètres change complètement la structure fine de l'IR ; dans le grave (grandes longueurs d'onde), beaucoup moins. D'où :

| Situation | Méthode | Réglages de départ |
|---|---|---|
| Piste rendue avec exactement cette IR (reverb à convolution) | **Inverse** | Régularisation −120 dB, Son direct 0 ms |
| Même chose, mais bruit ou queue de reverb coupée | **Itérative** | λ 0,01 |
| Prise classique, plusieurs instruments, un micro = une IR | **Salle**, préréglage *Ensemble / orchestre* | Garder 60 ms, grave exact désactivé |
| Soliste immobile dans une pièce qui sonne mal (guitare…) | **Salle**, préréglage *Soliste immobile* | Garder 15 ms, grave exact *Auto* |
| IR proche mais imparfaite, avec des sons secs représentatifs | **Neuro** (expérimental) | — |

### Prise classique, multi-micros

1. Placez un haut-parleur sur scène, à l'endroit du pupitre ou de l'instrument principal capté par chaque micro (ou au centre de l'ensemble).
2. Capture → *Exporter le signal…*, jouez-le et enregistrez **tous les micros en même temps** (un fichier multicanal ou un fichier par micro).
3. *Extraire les IR* : un fichier à plus de 2 canaux donne une IR par micro (`… · micro 1`, `… · micro 2`…). Une paire stéréo (AB, ORTF) reste une IR stéréo.
4. Pistes : assignez à chaque micro son IR, méthode **Salle**. Pour une paire stéréo, laissez *Canaux liés* coché : même gain sur les deux canaux, l'image stéréo est préservée.

La méthode Salle n'inverse pas l'IR : elle utilise son **enveloppe d'énergie par bande** (combien de réverbération arrive, à quel niveau, à quelle vitesse elle décroît), qui, elle, est quasiment la même partout dans la salle. Elle prédit à chaque instant la réverbération tardive contenue dans la piste et la retire, en gardant le son direct et les premières réflexions (*Garder les … ms*) pour rester naturelle. Augmentez *Force* et baissez *Atténuation max* pour un résultat plus sec, au risque d'artefacts.

À quoi s'attendre : sur une simulation très réverbérée où l'IR provient d'une autre position que la source, le rapport son direct / réverb tardive gagne environ 5 dB (test inclus). Retirer *toute* la réverbération d'un enregistrement d'ensemble est physiquement impossible avec une IR : chaque musicien a la sienne.

### Guitare (ou voix) dans une pièce qui sonne mal

Le problème d'une petite pièce, ce sont surtout les **modes dans le grave** (boum, résonances) et les premières réflexions. Ici on peut aller plus loin, car le musicien reste presque immobile :

1. Placez le haut-parleur **là où se trouvait la caisse de la guitare**, micro inchangé.
2. Faites **3 captures en déplaçant le haut-parleur de 3 à 5 cm** entre chaque (l'amplitude des mouvements du musicien).
3. Sélectionnez-les → *Combiner les positions*. Le logiciel mesure jusqu'à quelle fréquence les trois IR concordent (« Grave exact ≤ … Hz ») et moyenne le grave.
4. Pistes : méthode **Salle**, préréglage *Soliste immobile* (grave exact = *Auto*). Sous la fréquence mesurée, l'IR est **inversée exactement** — les modes de la pièce sont réellement corrigés, pas seulement atténués ; au-dessus, la réverbération est retirée statistiquement.

Sur une simulation de petite pièce (IR concordante jusqu'à 800 Hz), la partie grave passe de −17,6 dB à +13,5 dB de fidélité au signal sec.

## Les méthodes en détail

| Méthode | Principe |
|---|---|
| **Salle** | Prédiction de la réverbération tardive à partir de l'enveloppe temps-fréquence de l'IR (lissée en tiers d'octave), filtre de Wiener à décision dirigée ; option hybride : déconvolution exacte sous une fréquence de transition |
| **Inverse** | Déconvolution spectrale régularisée (Tikhonov) sur la piste entière, en une seule FFT |
| **Itérative** | Moindres carrés + a priori de parcimonie temps-fréquence, L-BFGS préconditionné par l'inverse, convolution linéaire exacte |
| **Neuro** | Réseau convolutif dilaté entraîné pour *une* IR (logique NAM) ; reçoit la piste et l'estimation inverse, prédit deux masques complexes |

Mesures sur signaux de synthèse (`python tests/test_core.py`, RT60 1,8 s) :

| Cas | Inverse | Itérative |
|---|---|---|
| IR exacte, sans bruit | 241 dB | — |
| Bruit à −50 dB | 34 dB | 40 dB |
| Queue de reverb coupée | 24 dB | 41 dB |

### Réglages importants

- **Son direct (ms)** — Inverse / Itérative avec une IR *mesurée* : 1 à 5 ms. Le résultat est l'instrument tel que le micro l'entend sans la salle. Sans cela, la courbe de réponse du haut-parleur de mesure serait inversée (un HP faible dans le grave donnerait un résultat boomy). 0 ms = impulsion parfaite, pour une IR numérique exacte.
- **Part directe / Gain de l'IR** : si la piste mélange sec et reverb (ex. 70 % dry / 30 % wet), renseignez ces niveaux — la piste est alors modélisée comme `dry·δ + wet·IR`.
- **Régularisation** : plus bas = plus exact mais plus sensible au bruit. −100 à −140 dB pour un rendu numérique exact, −30 à −50 dB pour un enregistrement réel.
- **Résidu de reconstruction** (Inverse / Itérative) : on reconvolue le résultat par l'IR et on compare à la piste. Sous −40 dB, l'IR explique bien la piste ; vers −10 dB, elle ne correspond pas (mauvaise IR, latence, gain, mix dry/wet). Pour la méthode Salle, l'indicateur est l'énergie du résultat par rapport à la piste.

### Capture

- Le **silence après chaque sweep** doit dépasser la durée de la reverb.
- Les sweeps sont moyennés (+3 dB de dynamique par doublement) ; la distorsion harmonique (jusqu'à l'ordre 8) est rejetée automatiquement.
- « Recaler sur le son direct » retire la latence d'enregistrement ; décochez-le pour conserver le pré-délai exact d'un plugin. Décochez « Normaliser » pour garder le gain absolu (utile si le niveau sec/reverb doit être exact).
- Les réglages sont aussi enregistrés dans `capture_sweep.sweep.json` et rechargés automatiquement si ce fichier est à côté de l'enregistrement.

### Mode neuronal (expérimental)

Comme NAM apprend un ampli, on apprend une reverb : des sons secs sont convolués à la volée avec l'IR (légèrement perturbée selon le réglage *Robustesse*) et le réseau apprend à retrouver le sec. Il démarre exactement sur le résultat de la méthode inverse et apprend à en corriger les erreurs.

- Sans sons secs (prises classiques, archives), ce mode n'est pas adapté : utilisez **Salle**.
- Fournissez un **dossier de sons secs proches de vos pistes** (voix sèches, batterie close-mic…) ; sans cela, un générateur de synthèse est utilisé, nettement moins représentatif.
- La courbe montre le SNR du modèle face à la méthode inverse seule (pointillés) sur les mêmes données de validation. **Si la courbe ne passe pas au-dessus, le modèle n'apporte rien** : l'application vous le signale.
- Comptez plusieurs milliers de pas sur GPU. Sur CPU, des essais courts (quelques centaines de pas) restent au niveau de la méthode inverse.
- Le modèle `.pt` embarque son IR : l'IR assignée à la piste est ignorée.

## Limites

- **Reverb « true stereo »** (IR à 4 canaux LL/LR/RL/RR) : non gérée ; les IR stéréo sont appliquées canal par canal.
- Reverbs **modulées ou non linéaires** (chorus dans la queue, saturation, gate) : une IR ne les décrit pas, le résultat sera approximatif.
- La dernière portion de piste (durée de l'IR) est moins précise si la queue de reverb a été coupée à l'export : exportez vos pistes avec la queue complète.

## Ligne de commande

```bash
python cli.py --sweep capture.wav
python cli.py --extract enregistrement.wav -o hall_IR.wav
python cli.py pistes/*.wav --ir hall_IR.wav --method room --late-ms 60 -o sorties/
python cli.py pistes/*.wav --ir hall_IR.wav --method iterative --reg -60 -o sorties/
python cli.py voix.wav --model hall.pt
```

## Structure

```
dereverb/core/   capture.py (sweeps, extraction IR, T30, cohérence)   statistical.py (Salle)
                 inverse.py   iterative.py
                 neural.py (réseau, entraînement, inférence)   engine.py   audio.py   synth.py
dereverb/gui/    app.py   page_tracks.py   page_capture.py   page_model.py   widgets.py   theme.py
```
