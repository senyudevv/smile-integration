# Avancement — 20 avril 2026

## Ce qui a été fait

### 1. Passage en français
Le projet était entièrement codé en italien. Les modifications suivantes ont été apportées :

**`src/utils/dialog_manager.py`**
- Prompt LLM principal traduit en français (persona Robot, règles de conversation, labels d'état GREETING/FREE_TALK/FAREWELL)
- Prompt de résumé de fin de session traduit en français
- `MODEL_NAME` corrigé : `"llama3"` → `"llama3:8b"` (nom réel du modèle installé)

**`src/utils/speech_utils.py`**
- `extract_name_from_text` : patterns italiens (`mi chiamo`, `sono`) remplacés par les équivalents français (`je m'appelle`, `je suis`, `c'est`)
- Blacklist et noms de fallback (`Utente_`) passés en français (`Utilisateur_`)
- Retour à l'auto-détection du microphone (`find_working_mic`) avec affichage du micro sélectionné

**`src/utils/profile_manager.py`**
- Labels historique : `Utente:` → `Utilisateur:`, message vide traduit

**`src/recognize_live.py`**
- Phrases parlées : accueil nouveau visage, reconnaissance, au revoir → français
- `greeting_keywords`, `farewell_keywords`, `goodbye_phrases` → équivalents français
- Messages console traduits en français
- Timeout Ollama passé à 120s (inférence CPU sans GPU)
- `max_silence_rounds` : 3 → 5 (plus de tolérance avant fin de conversation)
- Correction crash tracker : `try/except` autour de `tr.update(frame)`

**`src/utils/async_core.py`**
- Messages de démarrage et warm-up traduits en français

### 3. Installation et configuration

- `requirements.txt` réécrit en UTF-8 (fichier original encodé en UTF-16, illisible par pip)
- Dépendances installées sur Python 3.13 / CPU-only :
  - `audioop-lts` (audioop supprimé en Python 3.13)
  - `comtypes`, `cffi` (manquants pour pyttsx3 / vosk sur Windows)
  - `torch`, `torchvision` version CPU (`--index-url https://download.pytorch.org/whl/cpu`)
- Contournement `PYTHONPATH` pointant vers `C:\Users\Marwan\Documents\STAGE\packages` : `$env:PYTHONPATH = ""`
- `src/config.py` créé avec les chemins corrects pour cette machine
- Modèle Vosk français complet (`vosk-model-fr-0.22`) installé dans `C:/Users/LUTIN/models/`
- Chemin Vosk corrigé (modèle imbriqué d'un niveau : `vosk-model-fr-0.22/vosk-model-fr-0.22/`)
- Dossier renommé `Modèles` → `models` (accent causait une erreur Vosk)
- `config_example.py` mis à jour avec `VOSK_MODEL_PATH`, `MIC_INDEX`, `SILENCE_HANGOVER`
- Ollama installé avec `llama3:8b` (~5 Go)

### 4. État au soir du 20 avril

- ✅ Vosk charge et transcrit correctement en français
- ✅ TTS (pyttsx3) fonctionne et parle
- ✅ Reconnaissance faciale détecte les visages
- ✅ Ollama répond (vérifié : modèle présent, API accessible)
- ❌ Impossible d'avoir une conversation complète : le micro ne s'ouvre pas correctement (`[Errno -9999] Unanticipated host error` sur les index testés), les tours de parole s'enchaînent sans capturer l'audio

## À faire

- [ ] Résoudre le problème d'ouverture du microphone (tester les index PyAudio disponibles, vérifier les permissions audio Windows)
- [ ] Valider un échange complet : détection visage → salutation → question → réponse LLM → TTS
- [ ] Tester la qualité de transcription Vosk une fois le bon micro identifié
