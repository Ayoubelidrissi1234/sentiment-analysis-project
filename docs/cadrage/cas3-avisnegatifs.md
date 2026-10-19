# Fiche de cadrage : Priorisation des avis négatifs

## Besoin en une phrase
Détecter rapidement les avis négatifs des clients pour y répondre en priorité.

## Utilisateur final
Service client, équipe qualité et responsable expérience client.

## Données nécessaires
Avis textuels, notes, historique des réponses et résolutions.

## Approche retenue
Cocher une seule case :

- [ ] Règles métier
- [x] Machine learning
- [ ] RAG
- [ ] Modèle génératif seul

## Justification
(a) Nous avons beaucoup de données étiquetées, (b) la détection est mesurable et vérifiable.

## Métrique de succès
Précision > 90 %, rappel > 85 %.

## Conséquence d'une erreur
Avis manqué = client non satisfait ; fausse alerte = perte de temps.

## Risques éthiques
Ne pas censurer les avis ; traiter chaque client avec équité.
