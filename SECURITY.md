# Politique de sécurité

## Versions supportées

| Version | Support sécurité |
|---|---|
| 1.x (actuelle) | Oui |

## Signaler une vulnérabilité

**Ne pas ouvrir d'issue publique pour des vulnérabilités de sécurité.**

Envoyer un email à : **ibrahima.gabar.diop[at]sonatelacademy.sn**

Inclure :
- Description de la vulnérabilité
- Étapes pour la reproduire
- Impact potentiel
- Toute suggestion de correction

Une réponse sera apportée sous **72 heures**.

## Bonnes pratiques pour les contributeurs

- Ne jamais commiter de credentials, clés API ou mots de passe
- Le fichier `.env` est exclu du versioning — utiliser `.env.example` comme modèle
- `data/users.db` contient des données utilisateurs — ne jamais versionner
- `models/face_yolo.pt` — distribué via GitHub Releases uniquement

## Limitations connues

Ce projet est un **prototype académique**. Un déploiement en production nécessite :
- Un audit de sécurité complet
- Une authentification renforcée (2FA, sessions sécurisées)
- Le respect du RGPD et des législations locales sur la biométrie
- Le consentement explicite des personnes identifiées
