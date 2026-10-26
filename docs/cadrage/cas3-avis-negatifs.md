# Fiche de cadrage : Priorisation des avis négatifs

## Besoin en une phrase (obligatoire)
Détecter automatiquement les avis négatifs sur les réseaux sociaux pour répondre rapidement aux clients mécontents et protéger la réputation.

## Utilisateur final (obligatoire)
Le community manager et l'équipe communication qui traitent les avis clients sur les réseaux sociaux et les plateformes d'avis.

## Approche retenue (obligatoire)
Cocher une seule case :
- [ ] Règles métier
- [x] Machine learning
- [ ] RAG
- [ ] Modèle génératif seul

## Justification (obligatoire, citer au moins deux critères de la grille : (a) données étiquetées, (b) vérifiabilité, (c) coût par requête, (d) conséquence d'une erreur)
Nous retenons l'approche par machine learning car (a) nous disposons de données étiquetées d'avis et (c) le coût par requête doit rester faible pour traiter un grand volume.

## Données nécessaires et leur origine (obligatoire)
Avis clients collectés sur Trustpilot, Google Reviews et les réseaux sociaux, étiquetés comme positifs, négatifs ou neutres.

## Métrique de succès et seuil d'acceptation (obligatoire)
Rappel sur les avis négatifs (ne pas en rater), seuil minimum de 0.90 de rappel pour la classe négative.

## Conséquence d'une erreur et validation humaine prévue (obligatoire)
Un avis positif classé négatif est sans gravité, mais un avis négatif raté peut nuire à la réputation, donc un agent vérifie les cas limites.

## Risques éthiques ou de confidentialité (obligatoire)
Les avis publics sont moins sensibles mais il faut éviter toute discrimination et respecter les conditions d'utilisation des plateformes.

## Approche écartée et pourquoi (facultatif)
Le RAG est écarté car on n'a pas besoin de chercher dans des documents, juste classifier le sentiment.
