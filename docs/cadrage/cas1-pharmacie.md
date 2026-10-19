# Fiche de cadrage : Ordonnances incomplètes en pharmacie

## Besoin en une phrase
Détecter automatiquement les ordonnances manquantes d'informations essentielles pour prévenir les erreurs de délivrance.

## Utilisateur final
Les pharmaciens et leur équipe qui traitent chaque jour des centaines d'ordonnances.

## Données nécessaires
Ordonnances numérisées, base des médicaments autorisés, fiches patients, historique des ordonnances validées.

## Approche retenue
Cocher une seule case :

- [ ] Règles métier
- [x] Machine learning
- [ ] RAG
- [ ] Modèle génératif seul

## Justification
(a) Nous disposons de nombreuses données déjà annotées, (c) les règles métier deviendraient trop complexes à maintenir.

## Métrique de succès
Taux de détection supérieur à 95 %, taux de fausses alertes inférieur à 3 %.

## Conséquence d'une erreur
Faux positif = perte de temps pour l'équipe ; Faux négatif = risque pour la santé du patient.

## Risques éthiques
Protéger strictement les données de santé des patients ; garantir l'égalité de traitement.
