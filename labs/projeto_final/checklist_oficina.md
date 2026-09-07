# Checklist — oficina (aula 07)

Use nos 50 min de mentoria:

- [ ] Problema escrito em ≤5 linhas
- [ ] Dataset carregando (shape + target)
- [ ] Split feito **antes** de qualquer fit
- [ ] Baseline rodando e métrica anotada
- [ ] Rede neural rodando (mesmo que fraca)
- [ ] Tabela baseline × NN na validação
- [ ] Teste **ainda não** olhado (ou olhado só se modelo já escolhido)
- [ ] 3 limitações listadas
- [ ] README com `pip install` + comando de execução
- [ ] Slides da defesa (máx. 6)

Bloqueio comum: vazar informação do teste no pré-processamento. 
Se usar `StandardScaler`, faça `fit` só no treino.
