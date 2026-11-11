Verifica GA20: griglia di 10 giorni e raffinamento locale

full_grid_check.json: esecuzione integrata sulla finestra completa di due anni, con 6 processi e generazione dei grafici disattivata. Risultato: 13 missioni accettate dopo il controllo SOI, 4.413.263 Lambert, 48,86 secondi esclusi i grafici.

verified_candidate.json e all_13_soi_validation.json: ricerca automatica in una cella della griglia, con 763 cluster e 84 durate di ritorno. Nessun cluster preselezionato. Tutti i 13 candidati superano il controllo SOI.

missione_esempio.png: figura prodotta dal blocco grafico effettivo, con le epoche raffinate di un candidato verificato nel modello. Le distribuzioni orbitali sintetiche usate soltanto per il test di rendering non sono copiate qui.

La selezione preliminare usa un margine di 0,5 km/s e un budget di 64 semi per incontro, con tre nodi iniziali di TOF. Questi parametri definiscono una ricerca euristica; i vincoli fisici finali restano rigorosi. Le posizioni medie dei cluster vengono proiettate sulla superficie SOI per verificare la permanenza all'esterno della sfera lungo i due archi.
