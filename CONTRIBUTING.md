# Contribuer

## Workflow

1. Fork le repo
2. Créer une branche : `git checkout -b feat/ma-feature`
3. Commiter : `git commit -m "feat: description"`
4. Pousser : `git push origin feat/ma-feature`
5. Ouvrir une Pull Request

## Convention de commits

```
feat:     nouvelle fonctionnalité
fix:      correction de bug
docs:     documentation
refactor: refactoring sans changement fonctionnel
test:     ajout/modification de tests
chore:    maintenance
```

## Règles

- Tester localement avant de soumettre une PR
- Ne jamais commiter `.env`, `data/users.db`, ou `models/*.pt`
- Respecter la politique de sécurité ([SECURITY.md](SECURITY.md))
- Les vulnérabilités se signalent en privé, pas via les issues
