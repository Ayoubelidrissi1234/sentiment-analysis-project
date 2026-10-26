# Fiche de cadrage : Questions sur le règlement intérieur

## Besoin en une phrase (obligatoire)
Vérifier automatiquement la conformité des documents internes au règlement intérieur de l'entreprise et répondre aux questions des employés.

## Utilisateur final (obligatoire)
Le service RH et les employés qui cherchent des réponses rapides sur le règlement intérieur de l'entreprise.

## Approche retenue (obligatoire)
Cocher une seule case :
- [ ] Règles métier
- [ ] Machine learning
- [x] RAG
- [ ] Modèle génératif seul

## Justification (obligatoire, citer au moins deux critères de la grille : (a) données étiquetées, (b) vérifiabilité, (c) coût par requête, (d) conséquence d'une erreur)
Nous retenons l'approche RAG car (b) la vérifiabilité des réponses est essentielle et (d) une erreur sur le règlement a des conséquences importantes.

## Données nécessaires et leur origine (obligatoire)
Le règlement intérieur de l'entreprise, les conventions collectives et les documents RH internes fournis par le service juridique.

## Métrique de succès et seuil d'acceptation (obligatoire)
Taux de réponses correctes vérifiées par le service RH, seuil minimum de 90% de réponses acceptables.

## Conséquence d'une erreur et validation humaine prévue (obligatoire)
Une réponse erronée peut induire un employé en erreur, donc chaque réponse cite sa source et un agent RH valide les cas complexes.

## Risques éthiques ou de confidentialité (obligatoire)
Les documents internes sont confidentiels, le système ne doit pas exposer d'informations sensibles et doit tracer les accès.

## Approche écartée et pourquoi (facultatif)
Le machine learning seul est écarté car on manque de données étiquetées sur les questions de règlement.
