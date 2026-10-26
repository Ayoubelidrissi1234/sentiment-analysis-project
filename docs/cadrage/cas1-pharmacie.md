# Fiche de cadrage : Ordonnances incomplètes en pharmacie

## Besoin en une phrase (obligatoire)
Analyser automatiquement les avis clients de la pharmacie pour détecter les sentiments négatifs et améliorer la qualité du service.

## Utilisateur final (obligatoire)
Le pharmacien et son équipe qui souhaitent identifier rapidement les avis clients négatifs pour répondre et améliorer leur service.

## Approche retenue (obligatoire)
Cocher une seule case :
- [ ] Règles métier
- [x] Machine learning
- [ ] RAG
- [ ] Modèle génératif seul

## Justification (obligatoire, citer au moins deux critères de la grille : (a) données étiquetées, (b) vérifiabilité, (c) coût par requête, (d) conséquence d'une erreur)
Nous retenons l'approche par machine learning car (a) nous disposons de données étiquetées d'avis clients et (d) la conséquence d'une erreur est faible pour une pharmacie.

## Données nécessaires et leur origine (obligatoire)
Avis clients textuels collectés sur Google My Business et les pages de la pharmacie, étiquetés manuellement par l'équipe comme positifs ou négatifs.

## Métrique de succès et seuil d'acceptation (obligatoire)
F1-macro sur les avis négatifs, seuil minimum de 0.85 pour accepter le modèle en production.

## Conséquence d'une erreur et validation humaine prévue (obligatoire)
Une erreur de classification est sans conséquence grave, mais un pharmacien valide manuellement les avis signalés comme négatifs avant réponse.

## Risques éthiques ou de confidentialité (obligatoire)
Les avis contiennent parfois des données personnelles de patients, il faut anonymiser avant traitement et respecter le RGPD.

## Approche écartée et pourquoi (facultatif)
Le modèle génératif seul est écarté car trop coûteux par requête et moins vérifiable.
