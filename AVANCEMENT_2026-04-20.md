# Avancement — 20 avril 2026

## Ce qui a été fait

### 1. Documentation
- Création de `CLAUDE.md` à la racine du projet : architecture, commandes, structure des données, constantes de tuning.

### 2. Passage en français
Le projet était entièrement codé en italien. Les modifications suivantes ont été apportées :

**`src/utils/dialog_manager.py`**
- Prompt LLM principal traduit en français (persona Robot, règles de conversation, labels d'état GREETING/FREE_TALK/FAREWELL)
- Prompt de résumé de fin de session traduit en français
- Fallback mémoire épisodique : `"(nessuna memoria episodica disponibile)"` → `"(aucune mémoire épisodique disponible)"`

**`src/utils/speech_utils.py`**
- `extract_name_from_text` : patterns italiens (`mi chiamo`, `sono`) remplacés par les équivalents français (`je m'appelle`, `je suis`, `c'est`)
- Blacklist et noms de fallback (`Utente_`) passés en français (`Utilisateur_`)

**`src/utils/profile_manager.py`**
- Labels historique : `Utente:` → `Utilisateur:`, `Robot:` conservé
- Message d'historique vide traduit en français

**`src/recognize_live.py`**
- Phrases parlées : accueil nouveau visage, reconnaissance, au revoir → français
- `greeting_keywords` : `ciao`, `buongiorno`… → `bonjour`, `salut`, `bonsoir`…
- `farewell_keywords` et `goodbye_phrases` → équivalents français
- Prompt de fermeture de conversation traduit

### 3. Configuration
- `src/config_example.py` : ajout de `VOSK_MODEL_PATH` pointant vers un modèle Vosk français
- `src/config.py` créé avec les chemins corrects pour cette machine (`C:/Users/LUTIN/...`)
- `VOSK_MODEL_PATH` mis à jour vers `C:/Users/LUTIN/Modèles/vosk-model-small-fr-0.22` (modèle léger installé)

## En cours / À faire

- [ ] Modèle Vosk complet (`vosk-model-fr-0.22`) en cours de téléchargement — mettre à jour `VOSK_MODEL_PATH` une fois téléchargé
- [ ] Installer les dépendances Python : problème `PYTHONPATH` pointant vers `C:\Users\Marwan\Documents\STAGE\packages` (machine partagée) — fix : `$env:PYTHONPATH = ""` puis `pip install -r requirements.txt`
- [ ] Tester le lancement complet (`ollama serve` + `python -m src.recognize_live`)
- [ ] Vérifier la voix TTS en français (tester `DEFAULT_VOICE_INDEX = 0` ou `1`)
