/**
 * Domande del Test
 * 
 * Ogni domanda ha un ID univoco.
 * Le opzioni sono frasi descrittive che vanno direttamente nell'embedding.
 */

export type Section = "psychological" | "values" | "interests" | "behavioral";

export type Question = 
  | { id: string; type: "closed"; text: string; options: string[] }
  | { id: string; type: "open"; text: string; placeholder?: string };

export const QUESTIONS: Record<Section, Question[]> = {

  // ==========================================
  // PSYCHOLOGICAL (personalità e tratti)
  // ==========================================
  psychological: [
    {
      id: "psy-1",
      type: "closed",
      text: "Come ti senti in situazioni sociali con persone nuove?",
      options: [
        "Mi sento a disagio con persone nuove, preferisco evitare",
        "Sono un po' nervoso/a all'inizio ma mi adatto",
        "Dipende dalla situazione e dalle persone",
        "Mi sento abbastanza a mio agio con persone nuove",
        "Amo conoscere gente nuova, mi viene naturale",
      ],
    },
    {
      id: "psy-2",
      type: "closed",
      text: "Dopo una giornata intensa, come preferisci ricaricarti?",
      options: [
        "Ho bisogno di stare completamente solo/a in silenzio",
        "Preferisco rilassarmi tranquillamente a casa",
        "Dipende da come mi sento",
        "Mi piace stare con poche persone fidate",
        "Mi ricarico uscendo e socializzando",
      ],
    },
    {
      id: "psy-3",
      type: "closed",
      text: "Quanto ti piace provare esperienze nuove?",
      options: [
        "Preferisco le mie routine e abitudini consolidate",
        "Provo cose nuove solo se necessario",
        "Sono aperto/a al nuovo ma con cautela",
        "Mi piace sperimentare cose diverse",
        "Cerco sempre nuove esperienze e stimoli",
      ],
    },
    {
      id: "psy-4",
      type: "closed",
      text: "Come reagisci ai cambiamenti improvvisi?",
      options: [
        "I cambiamenti mi destabilizzano molto",
        "Faccio fatica ad adattarmi ai cambiamenti",
        "Mi adatto con un po' di tempo",
        "Gestisco bene i cambiamenti",
        "Amo i cambiamenti, mi stimolano",
      ],
    },
    {
      id: "psy-5",
      type: "closed",
      text: "Come gestisci lo stress e le situazioni difficili?",
      options: [
        "Mi agito molto e faccio fatica a gestirlo",
        "Mi preoccupo parecchio ma cerco di farcela",
        "Ho alti e bassi nella gestione dello stress",
        "Resto abbastanza calmo/a sotto pressione",
        "Sono molto stabile anche nelle difficoltà",
      ],
    },
    {
      id: "psy-6",
      type: "closed",
      text: "Quanto percepisci le emozioni delle persone intorno a te?",
      options: [
        "Faccio fatica a capire cosa provano gli altri",
        "Noto le emozioni solo se sono molto evidenti",
        "Percepisco le emozioni degli altri abbastanza",
        "Sono piuttosto empatico/a e sensibile",
        "Sento profondamente le emozioni degli altri",
      ],
    },
    {
      id: "psy-7",
      type: "closed",
      text: "Come prendi le decisioni importanti?",
      options: [
        "Decido di impulso senza pensarci troppo",
        "Tendo a decidere velocemente",
        "Bilancio intuizione e ragionamento",
        "Preferisco riflettere bene prima di decidere",
        "Analizzo tutto a fondo prima di ogni decisione",
      ],
    },
    {
      id: "psy-8",
      type: "closed",
      text: "Come ti comporti in gruppo?",
      options: [
        "Preferisco ascoltare e stare in disparte",
        "Intervengo solo se interpellato/a",
        "Partecipo quando mi sento a mio agio",
        "Sono spesso attivo/a nelle discussioni",
        "Tendo naturalmente a guidare il gruppo",
      ],
    },
    {
      id: "psy-9",
      type: "closed",
      text: "Quanto sei organizzato/a nella vita quotidiana?",
      options: [
        "Sono molto spontaneo/a, poca organizzazione",
        "Tendo ad essere disordinato/a",
        "Sono moderatamente organizzato/a",
        "Sono abbastanza metodico/a e ordinato/a",
        "Sono molto organizzato/a e pianificato/a",
      ],
    },
    {
      id: "psy-open",
      type: "open",
      text: "Come ti descriveresti in poche parole?",
      placeholder: "Es: Sono una persona curiosa e riflessiva che...",
    },
  ],

  // ==========================================
  // VALUES (valori e priorità di vita)
  // ==========================================
  values: [
    {
      id: "val-1",
      type: "closed",
      text: "Quanto è importante la famiglia nella tua vita?",
      options: [
        "Sono molto indipendente, la famiglia non è centrale",
        "La famiglia è importante ma non prioritaria",
        "Bilancio famiglia e vita personale",
        "La famiglia è molto importante per me",
        "La famiglia è la mia priorità assoluta",
      ],
    },
    {
      id: "val-2",
      type: "closed",
      text: "Che ruolo ha il lavoro nella tua vita?",
      options: [
        "Il lavoro è solo un mezzo, non mi definisce",
        "Lavoro per vivere, non vivo per lavorare",
        "Cerco equilibrio tra carriera e vita privata",
        "La carriera è importante per la mia realizzazione",
        "Sono molto ambizioso/a, il successo è prioritario",
      ],
    },
    {
      id: "val-3",
      type: "closed",
      text: "Quanto è importante l'onestà per te?",
      options: [
        "A volte una bugia bianca è necessaria",
        "Sono onesto/a ma uso diplomazia",
        "L'onestà è importante ma con tatto",
        "Sono molto diretto/a e sincero/a",
        "L'onestà totale è un valore fondamentale",
      ],
    },
    {
      id: "val-4",
      type: "closed",
      text: "Cosa preferisci: stabilità o avventura?",
      options: [
        "Cerco sicurezza e stabilità sopra tutto",
        "Preferisco la stabilità con qualche novità",
        "Mi piace un equilibrio tra i due",
        "Amo l'avventura pur avendo una base stabile",
        "Vivo per l'avventura e le novità",
      ],
    },
    {
      id: "val-5",
      type: "closed",
      text: "Quanto contano i soldi per te?",
      options: [
        "I soldi non sono importanti, basta il necessario",
        "La sicurezza economica basta",
        "Voglio vivere bene senza eccessi",
        "L'agiatezza economica è importante",
        "Il successo finanziario è molto importante",
      ],
    },
    {
      id: "val-6",
      type: "closed",
      text: "Quanto è importante la crescita personale?",
      options: [
        "Sono contento/a di come sono",
        "Miglioro se serve",
        "Cerco di crescere quando posso",
        "La crescita personale è importante per me",
        "Sono sempre alla ricerca di migliorarmi",
      ],
    },
    {
      id: "val-7",
      type: "closed",
      text: "Che importanza dai alla spiritualità o religione?",
      options: [
        "Non sono interessato/a alla spiritualità",
        "Sono laico/a ma rispetto chi crede",
        "Ho una mia spiritualità personale",
        "La spiritualità è parte della mia vita",
        "La fede è centrale nella mia esistenza",
      ],
    },
    {
      id: "val-8",
      type: "closed",
      text: "Quanto è importante aiutare gli altri?",
      options: [
        "Mi concentro principalmente su me stesso/a",
        "Aiuto se posso senza troppo impegno",
        "Mi piace aiutare quando ne ho l'occasione",
        "Aiutare gli altri mi dà soddisfazione",
        "Dedico molto tempo ad aiutare gli altri",
      ],
    },
    {
      id: "val-9",
      type: "closed",
      text: "Come vedi le tradizioni?",
      options: [
        "Le tradizioni sono superate, guardo al futuro",
        "Rispetto le tradizioni ma non mi vincolano",
        "Alcune tradizioni sono importanti per me",
        "Le tradizioni familiari sono molto importanti",
        "Tengo molto alle tradizioni e ai valori classici",
      ],
    },
    {
      id: "val-open",
      type: "open",
      text: "Qual è il principio che guida le tue scelte importanti?",
      placeholder: "Es: Cerco sempre di essere autentico e...",
    },
  ],

  // ==========================================
  // INTERESTS (hobby e stile di vita)
  // ==========================================
  interests: [
    {
      id: "int-1",
      type: "closed",
      text: "Quanto sei attivo/a fisicamente?",
      options: [
        "Preferisco attività tranquille, poco sport",
        "Faccio poca attività fisica",
        "Sono moderatamente attivo/a",
        "Faccio sport regolarmente",
        "Sono molto sportivo/a, mi alleno spesso",
      ],
    },
    {
      id: "int-2",
      type: "closed",
      text: "Come preferisci trascorrere il tempo libero?",
      options: [
        "Amo stare a casa da solo/a",
        "Preferisco attività tranquille e solitarie",
        "Alterno momenti da solo/a e in compagnia",
        "Mi piace fare cose con amici",
        "Amo sempre stare in compagnia",
      ],
    },
    {
      id: "int-3",
      type: "closed",
      text: "Quanto ti piace viaggiare?",
      options: [
        "Preferisco stare a casa, non amo viaggiare",
        "Viaggio raramente, solo se necessario",
        "Mi piace viaggiare ogni tanto",
        "Adoro viaggiare e scoprire posti nuovi",
        "Viaggerei sempre, è la mia passione",
      ],
    },
    {
      id: "int-4",
      type: "closed",
      text: "Quanto ti interessa l'arte e la cultura?",
      options: [
        "Non mi interessano mostre, musei, teatro",
        "Ci vado raramente",
        "Apprezzo la cultura occasionalmente",
        "Mi piace andare a eventi culturali",
        "Arte e cultura sono le mie passioni",
      ],
    },
    {
      id: "int-5",
      type: "closed",
      text: "Che rapporto hai con la tecnologia?",
      options: [
        "Uso la tecnologia il minimo indispensabile",
        "La uso per necessità ma non mi appassiona",
        "Ho un rapporto equilibrato con la tecnologia",
        "Mi piace la tecnologia e le novità tech",
        "Sono appassionato/a di tecnologia",
      ],
    },
    {
      id: "int-6",
      type: "closed",
      text: "Ti piace cucinare?",
      options: [
        "Non cucino mai, non mi interessa",
        "Cucino solo il necessario",
        "Cucino abbastanza, mi arrangio bene",
        "Mi piace cucinare per me e gli altri",
        "Cucinare è una mia grande passione",
      ],
    },
    {
      id: "int-7",
      type: "closed",
      text: "Quanto ami la natura e le attività all'aperto?",
      options: [
        "Preferisco stare in città, la natura non fa per me",
        "Ogni tanto apprezzo una passeggiata",
        "Mi piace la natura di tanto in tanto",
        "Amo stare nella natura appena posso",
        "La natura è il mio elemento, ci vivo",
      ],
    },
    {
      id: "int-8",
      type: "closed",
      text: "Ti piace leggere?",
      options: [
        "Non leggo mai libri",
        "Leggo raramente",
        "Leggo ogni tanto quando trovo qualcosa",
        "Leggo regolarmente, mi piace",
        "Sono un/a lettore/trice appassionato/a",
      ],
    },
    {
      id: "int-9",
      type: "closed",
      text: "Che rapporto hai con la musica?",
      options: [
        "La musica non è importante per me",
        "Ascolto musica di sottofondo ogni tanto",
        "Mi piace la musica, l'ascolto spesso",
        "La musica è importante nella mia vita",
        "Vivo di musica, suono o canto",
      ],
    },
    {
      id: "int-open",
      type: "open",
      text: "Quali sono le tue passioni e cosa ti rende felice?",
      placeholder: "Es: Amo il trekking, la fotografia, cucinare...",
    },
  ],

  // ==========================================
  // BEHAVIORAL (stile relazionale)
  // ==========================================
  behavioral: [
    {
      id: "beh-1",
      type: "closed",
      text: "Quanto spesso ti piace comunicare con chi frequenti?",
      options: [
        "Preferisco messaggiare poco, quando serve",
        "Mi basta sentirci ogni tanto",
        "Mi piace una comunicazione regolare",
        "Mi piace sentirci spesso durante il giorno",
        "Amo comunicare costantemente",
      ],
    },
    {
      id: "beh-2",
      type: "closed",
      text: "Che tipo di conversazioni preferisci?",
      options: [
        "Preferisco conversazioni leggere e spensierate",
        "Mi trovo meglio con chiacchiere informali",
        "Alterno conversazioni leggere e profonde",
        "Preferisco conversazioni significative",
        "Cerco sempre conversazioni profonde e intime",
      ],
    },
    {
      id: "beh-3",
      type: "closed",
      text: "Come affronti i conflitti in una relazione?",
      options: [
        "Evito i conflitti il più possibile",
        "Preferisco lasciar passare le cose",
        "Affronto i problemi quando necessario",
        "Preferisco parlare e chiarire apertamente",
        "Affronto sempre i problemi subito e direttamente",
      ],
    },
    {
      id: "beh-4",
      type: "closed",
      text: "Quanto tempo hai bisogno per aprirti con qualcuno?",
      options: [
        "Mi apro molto lentamente, ci vuole tempo",
        "Sono riservato/a, mi apro gradualmente",
        "Dipende dalla persona e dalla situazione",
        "Mi apro abbastanza facilmente",
        "Sono un libro aperto fin da subito",
      ],
    },
    {
      id: "beh-5",
      type: "closed",
      text: "Quanto è importante per te il tempo da solo/a in una relazione?",
      options: [
        "Ho bisogno di molto tempo per me stesso/a",
        "Ho bisogno di spazi miei regolari",
        "Bilancio tempo insieme e tempo per me",
        "Preferisco passare molto tempo insieme",
        "Vorrei stare sempre insieme alla persona amata",
      ],
    },
    {
      id: "beh-6",
      type: "closed",
      text: "Come dimostri affetto?",
      options: [
        "Non sono molto espansivo/a, lo dimostro con i fatti",
        "Preferisco gesti concreti alle parole",
        "Uso sia parole che gesti",
        "Sono affettuoso/a e lo dimostro spesso",
        "Sono molto espansivo/a e fisico/a",
      ],
    },
    {
      id: "beh-7",
      type: "closed",
      text: "Quanto sei flessibile sui piani?",
      options: [
        "Ho bisogno di pianificare tutto in anticipo",
        "Preferisco avere piani definiti",
        "Sono flessibile ma apprezzo un po' di struttura",
        "Mi adatto facilmente ai cambiamenti di programma",
        "Sono molto spontaneo/a, vivo alla giornata",
      ],
    },
    {
      id: "beh-8",
      type: "closed",
      text: "Come gestisci le critiche?",
      options: [
        "Le critiche mi feriscono molto",
        "Faccio fatica ad accettare le critiche",
        "Ascolto le critiche e ci rifletto",
        "Accetto le critiche costruttive positivamente",
        "Apprezzo le critiche, mi aiutano a crescere",
      ],
    },
    {
      id: "beh-9",
      type: "closed",
      text: "Quanto ti piace fare progetti per il futuro?",
      options: [
        "Vivo nel presente, non faccio piani",
        "Preferisco non pianificare troppo avanti",
        "Faccio qualche progetto ma resto flessibile",
        "Mi piace avere obiettivi e progetti",
        "Amo pianificare e avere una visione chiara del futuro",
      ],
    },
    {
      id: "beh-open",
      type: "open",
      text: "Come ti comporti quando conosci qualcuno che ti interessa?",
      placeholder: "Es: All'inizio sono riservato/a ma poi...",
    },
  ],
};

// ============================================
// TEST INFO
// ============================================

/** Nome/versione del test corrente */
export const TEST_NAME = "personality-v1";

export const SECTIONS: Section[] = ["psychological", "values", "interests", "behavioral"];

export function getQuestionById(id: string): Question | undefined {
  for (const section of SECTIONS) {
    const q = QUESTIONS[section].find(q => q.id === id);
    if (q) return q;
  }
  return undefined;
}

export const TOTAL_QUESTIONS = Object.values(QUESTIONS).flat().length;
