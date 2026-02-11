/**
 * Definition of all assessment questions.
 */

export type Section = "psychological" | "values" | "interests" | "behavioral";

export type ClosedQuestion = {
  id: string;
  type: "closed";
  text: string; // Statement to be evaluated
  options: [string, string, string, string, string]; // Sentences for embedding (1-5)
  scaleLabels: [string, string]; // [min label, max label] e.g.: ["Never", "Always"]
};

export type OpenQuestion = {
  id: string;
  type: "open";
  text: string;
  template: string; // {answer} is replaced with the response
  placeholder?: string;
};

export type Question = ClosedQuestion | OpenQuestion;

export const QUESTIONS: Record<Section, Question[]> = {

  // ==========================================
  // PSYCHOLOGICAL (personality and traits)
  // ==========================================
  psychological: [
    {
      id: "psy-1",
      type: "closed",
      text: "Mi sento a mio agio in situazioni sociali con persone nuove",
      scaleLabels: ["Mai", "Sempre"],
      options: [
        "In situazioni sociali con sconosciuti, mi sento a disagio e preferisco evitare",
        "Quando conosco persone nuove, sono spesso nervoso",
        "Con le persone nuove, il mio comfort varia in base alla situazione",
        "Mi trovo abbastanza a mio agio quando conosco persone nuove",
        "Mi sento sempre a mio agio in situazioni sociali, socializzare mi viene naturale",
      ],
    },
    {
      id: "psy-2",
      type: "closed",
      text: "Ho bisogno di stare da solo per ricaricarmi dopo attività sociali",
      scaleLabels: ["Mai", "Sempre"],
      options: [
        "Mi ricarico uscendo e socializzando, non ho bisogno di solitudine",
        "Preferisco stare con altri per rilassarmi",
        "A volte ho bisogno di stare solo, altre volte no",
        "Spesso ho bisogno di solitudine per ricaricarmi",
        "Ho sempre bisogno di stare completamente solo in silenzio per ricaricarmi",
      ],
    },
    {
      id: "psy-3",
      type: "closed",
      text: "Mi piace provare esperienze e cose nuove",
      scaleLabels: ["Per niente", "Moltissimo"],
      options: [
        "Preferisco le mie routine consolidate e non amo le novità",
        "Provo cose nuove solo quando è strettamente necessario",
        "Sono moderatamente aperto alle novità",
        "Mi piace sperimentare cose diverse regolarmente",
        "Cerco sempre nuove esperienze, la routine mi annoia",
      ],
    },
    {
      id: "psy-4",
      type: "closed",
      text: "Mi adatto facilmente ai cambiamenti improvvisi",
      scaleLabels: ["Per niente", "Molto bene"],
      options: [
        "I cambiamenti improvvisi mi destabilizzano molto",
        "Faccio fatica ad adattarmi ai cambiamenti",
        "Mi adatto ai cambiamenti con un po' di tempo",
        "Gestisco bene i cambiamenti improvvisi",
        "Mi adatto perfettamente ai cambiamenti, mi stimolano",
      ],
    },
    {
      id: "psy-5",
      type: "closed",
      text: "Riesco a mantenere la calma nelle situazioni stressanti",
      scaleLabels: ["Mai", "Sempre"],
      options: [
        "Lo stress mi travolge facilmente, perdo la calma",
        "Faccio fatica a gestire lo stress, mi preoccupo molto",
        "Ho alti e bassi nella gestione dello stress",
        "Resto generalmente calmo sotto pressione",
        "Sono sempre stabile emotivamente anche nelle situazioni difficili",
      ],
    },
    {
      id: "psy-6",
      type: "closed",
      text: "Percepisco facilmente le emozioni delle persone intorno a me",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Faccio fatica a capire cosa provano gli altri",
        "Noto le emozioni degli altri solo quando sono molto evidenti",
        "Percepisco le emozioni degli altri in modo normale",
        "Sono piuttosto empatico e sensibile alle emozioni altrui",
        "Sento profondamente le emozioni degli altri, sono molto empatico",
      ],
    },
    {
      id: "psy-7",
      type: "closed",
      text: "Prima di prendere decisioni importanti, rifletto a lungo",
      scaleLabels: ["Mai", "Sempre"],
      options: [
        "Decido di impulso seguendo l'istinto senza pensarci",
        "Tendo a decidere velocemente fidandomi dell'intuizione",
        "Bilancio intuizione e ragionamento nelle mie decisioni",
        "Preferisco riflettere bene prima di prendere decisioni",
        "Analizzo sempre tutto a fondo prima di ogni decisione",
      ],
    },
    {
      id: "psy-8",
      type: "closed",
      text: "In un gruppo, tendo a prendere l'iniziativa",
      scaleLabels: ["Mai", "Sempre"],
      options: [
        "In gruppo preferisco ascoltare e stare in disparte",
        "Intervengo solo quando vengo interpellato direttamente",
        "Partecipo quando mi sento a mio agio",
        "Sono spesso attivo nelle discussioni e propongo idee",
        "Tendo naturalmente a guidare il gruppo e prendere l'iniziativa",
      ],
    },
    {
      id: "psy-9",
      type: "closed",
      text: "Sono una persona organizzata e metodica",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Sono molto spontaneo e disorganizzato",
        "Tendo ad essere disordinato e poco metodico",
        "Sono moderatamente organizzato",
        "Sono abbastanza metodico e ordinato nella vita quotidiana",
        "Sono molto organizzato e pianificato, ogni cosa ha il suo posto",
      ],
    },
    {
      id: "psy-open",
      type: "open",
      text: "How would you describe yourself in a few words?",
      template: "I describe myself as a {answer} person",
      placeholder: "curious, reflective, sociable, creative...",
    },
  ],

  // ==========================================
  // VALUES (life values and priorities)
  // ==========================================
  values: [
    {
      id: "val-1",
      type: "closed",
      text: "La famiglia è una priorità importante nella mia vita",
      scaleLabels: ["Per niente", "Assolutamente"],
      options: [
        "Sono molto indipendente, la famiglia non è centrale nella mia vita",
        "La famiglia è importante ma non è la mia priorità principale",
        "Cerco un buon equilibrio tra famiglia e vita personale",
        "La famiglia è molto importante per me e influenza le mie scelte",
        "La famiglia è la mia priorità assoluta, viene prima di tutto",
      ],
    },
    {
      id: "val-2",
      type: "closed",
      text: "Il successo professionale è importante per me",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Il lavoro è solo un mezzo per vivere, non mi definisce",
        "Lavoro per vivere, non vivo per lavorare",
        "Cerco equilibrio tra carriera e vita privata",
        "La carriera è importante per la mia realizzazione personale",
        "Sono molto ambizioso, il successo professionale è una priorità",
      ],
    },
    {
      id: "val-3",
      type: "closed",
      text: "Preferisco sempre dire la verità, anche se scomoda",
      scaleLabels: ["Per niente", "Assolutamente"],
      options: [
        "Penso che a volte una bugia bianca sia necessaria",
        "Sono onesto ma uso molta diplomazia",
        "L'onestà è importante per me, ma la esprimo con tatto",
        "Sono molto diretto e sincero",
        "L'onestà totale è un valore fondamentale per me",
      ],
    },
    {
      id: "val-4",
      type: "closed",
      text: "Preferisco l'avventura e le novità rispetto alla stabilità",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Cerco sicurezza e stabilità sopra ogni altra cosa",
        "Preferisco la stabilità con qualche piccola novità",
        "Mi piace un buon equilibrio tra stabilità e avventura",
        "Amo l'avventura e le novità",
        "Vivo per l'avventura e le esperienze nuove",
      ],
    },
    {
      id: "val-5",
      type: "closed",
      text: "Il benessere economico è importante per la mia felicità",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "I soldi non sono importanti per me, basta il necessario",
        "Mi basta la sicurezza economica di base",
        "Voglio vivere bene senza eccessi",
        "L'agiatezza economica è importante per la mia vita",
        "Il successo finanziario è molto importante per me",
      ],
    },
    {
      id: "val-6",
      type: "closed",
      text: "Dedico tempo e energie alla mia crescita personale",
      scaleLabels: ["Mai", "Sempre"],
      options: [
        "Sono contento di come sono, non sento bisogno di cambiare",
        "Mi miglioro quando serve, ma non è una priorità",
        "Cerco di crescere quando ne ho l'occasione",
        "La crescita personale è importante, investo tempo in me stesso",
        "Sono sempre alla ricerca di migliorarmi",
      ],
    },
    {
      id: "val-7",
      type: "closed",
      text: "La spiritualità o religione è importante nella mia vita",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Non sono interessato alla spiritualità o alla religione",
        "Sono laico ma rispetto le credenze degli altri",
        "Ho una mia spiritualità personale",
        "La spiritualità è una parte importante della mia vita",
        "La fede è centrale nella mia esistenza",
      ],
    },
    {
      id: "val-8",
      type: "closed",
      text: "Aiutare gli altri mi dà grande soddisfazione",
      scaleLabels: ["Per niente", "Moltissimo"],
      options: [
        "Mi concentro principalmente su me stesso e i miei obiettivi",
        "Aiuto gli altri quando posso, senza troppo impegno",
        "Mi piace aiutare quando ne ho l'occasione",
        "Aiutare gli altri mi dà grande soddisfazione",
        "Dedico molto del mio tempo ad aiutare gli altri",
      ],
    },
    {
      id: "val-9",
      type: "closed",
      text: "Le tradizioni e i valori classici sono importanti per me",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Le tradizioni sono superate, guardo solo al futuro",
        "Rispetto le tradizioni ma non mi sento vincolato",
        "Alcune tradizioni sono importanti per me",
        "Le tradizioni familiari e culturali sono molto importanti",
        "Tengo molto alle tradizioni e ai valori classici",
      ],
    },
    {
      id: "val-open",
      type: "open",
      text: "What is the principle that guides your important choices?",
      template: "The principle that guides my choices is {answer}",
      placeholder: "authenticity, respect, freedom, family...",
    },
  ],

  // ==========================================
  // INTERESTS (hobbies and lifestyle)
  // ==========================================
  interests: [
    {
      id: "int-1",
      type: "closed",
      text: "Faccio attività fisica regolarmente",
      scaleLabels: ["Mai", "Ogni giorno"],
      options: [
        "Non faccio attività fisica, lo sport non mi interessa",
        "Faccio poca attività fisica, solo quando necessario",
        "Sono moderatamente attivo, faccio movimento regolarmente",
        "Faccio sport regolarmente e mi tengo in forma",
        "Sono molto sportivo, l'attività fisica è parte della mia routine quotidiana",
      ],
    },
    {
      id: "int-2",
      type: "closed",
      text: "Preferisco passare il tempo libero in compagnia",
      scaleLabels: ["Mai", "Sempre"],
      options: [
        "Amo stare a casa da solo, la solitudine mi rilassa",
        "Preferisco attività tranquille e solitarie",
        "Alterno momenti da solo e momenti in compagnia",
        "Mi piace fare attività con amici",
        "Amo sempre stare in compagnia, non mi piace stare da solo",
      ],
    },
    {
      id: "int-3",
      type: "closed",
      text: "Viaggiare e scoprire posti nuovi mi appassiona",
      scaleLabels: ["Per niente", "Moltissimo"],
      options: [
        "Preferisco stare a casa, viaggiare non mi attrae",
        "Viaggio raramente, solo quando è necessario",
        "Mi piace viaggiare ogni tanto per vacanza",
        "Adoro viaggiare e scoprire posti nuovi",
        "Viaggerei sempre se potessi, esplorare è la mia passione",
      ],
    },
    {
      id: "int-4",
      type: "closed",
      text: "Mi interessano l'arte, i musei e gli eventi culturali",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Non mi interessano mostre, musei o eventi culturali",
        "Vado a eventi culturali raramente",
        "Apprezzo la cultura occasionalmente",
        "Mi piace frequentare eventi culturali e artistici",
        "Arte e cultura sono le mie passioni",
      ],
    },
    {
      id: "int-5",
      type: "closed",
      text: "Sono appassionato di tecnologia e innovazione",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Uso la tecnologia il minimo indispensabile",
        "La uso per necessità ma non mi interessa",
        "Ho un rapporto equilibrato con la tecnologia",
        "Mi piace la tecnologia e sono aggiornato sulle novità",
        "Sono appassionato di tecnologia, è uno dei miei interessi principali",
      ],
    },
    {
      id: "int-6",
      type: "closed",
      text: "Cucinare è un'attività che mi piace",
      scaleLabels: ["Per niente", "Moltissimo"],
      options: [
        "Non cucino mai, la cucina non mi interessa",
        "Cucino solo il minimo necessario",
        "Cucino abbastanza bene e me la cavo",
        "Mi piace cucinare per me e per gli altri",
        "Cucinare è una mia grande passione",
      ],
    },
    {
      id: "int-7",
      type: "closed",
      text: "Amo stare nella natura e fare attività all'aperto",
      scaleLabels: ["Per niente", "Moltissimo"],
      options: [
        "Preferisco stare in città, la natura non fa per me",
        "Apprezzo una passeggiata ogni tanto",
        "Mi piace stare nella natura di tanto in tanto",
        "Amo stare nella natura e ci vado spesso",
        "La natura è il mio elemento, passo molto tempo all'aria aperta",
      ],
    },
    {
      id: "int-8",
      type: "closed",
      text: "Leggere libri è una delle mie attività preferite",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Non leggo mai libri",
        "Leggo raramente",
        "Leggo ogni tanto",
        "Leggo regolarmente",
        "Sono un lettore appassionato, leggo costantemente",
      ],
    },
    {
      id: "int-9",
      type: "closed",
      text: "La musica è una parte importante della mia vita",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "La musica non è importante per me",
        "Ascolto musica di sottofondo ogni tanto",
        "Mi piace la musica, l'ascolto spesso",
        "La musica è importante nella mia vita",
        "Vivo di musica, suono o canto attivamente",
      ],
    },
    {
      id: "int-open",
      type: "open",
      text: "What are your passions and what makes you happy?",
      template: "My passions are {answer}",
      placeholder: "traveling, cooking, photography, sports...",
    },
  ],

  // ==========================================
  // BEHAVIORAL (relational style)
  // ==========================================
  behavioral: [
    {
      id: "beh-1",
      type: "closed",
      text: "Mi piace comunicare spesso con le persone a cui tengo",
      scaleLabels: ["Per niente", "Moltissimo"],
      options: [
        "Preferisco comunicare poco, solo quando necessario",
        "Mi basta sentire le persone ogni tanto",
        "Mi piace una comunicazione regolare ma non eccessiva",
        "Mi piace sentire spesso le persone durante il giorno",
        "Amo comunicare costantemente con chi mi sta a cuore",
      ],
    },
    {
      id: "beh-2",
      type: "closed",
      text: "Preferisco conversazioni profonde e significative",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Preferisco conversazioni leggere e spensierate",
        "Mi trovo meglio con chiacchiere informali",
        "Alterno conversazioni leggere e profonde",
        "Preferisco conversazioni significative e con sostanza",
        "Cerco sempre conversazioni profonde e intime",
      ],
    },
    {
      id: "beh-3",
      type: "closed",
      text: "Affronto i conflitti direttamente invece di evitarli",
      scaleLabels: ["Mai", "Sempre"],
      options: [
        "Evito i conflitti il più possibile",
        "Preferisco lasciar passare le cose piuttosto che affrontarle",
        "Affronto i problemi quando è davvero necessario",
        "Preferisco parlare e chiarire apertamente",
        "Affronto sempre i problemi subito e direttamente",
      ],
    },
    {
      id: "beh-4",
      type: "closed",
      text: "Mi apro facilmente con le persone nuove",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Mi apro molto lentamente, ci vuole molto tempo",
        "Sono riservato, mi apro gradualmente",
        "Dipende dalla persona",
        "Mi apro abbastanza facilmente",
        "Sono un libro aperto fin da subito",
      ],
    },
    {
      id: "beh-5",
      type: "closed",
      text: "Ho bisogno di tempo per me stesso anche in una relazione",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Vorrei stare sempre insieme alla persona che amo",
        "Preferisco passare molto tempo insieme",
        "Cerco equilibrio tra tempo insieme e tempo per me",
        "Ho bisogno di spazi miei regolari",
        "Ho bisogno di molto tempo per me stesso",
      ],
    },
    {
      id: "beh-6",
      type: "closed",
      text: "Sono una persona affettuosa e lo dimostro apertamente",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Non sono espansivo, dimostro affetto con i fatti",
        "Preferisco gesti concreti alle parole",
        "Uso sia parole che gesti per esprimere affetto",
        "Sono affettuoso e lo dimostro spesso",
        "Sono molto espansivo e fisico nel dimostrare affetto",
      ],
    },
    {
      id: "beh-7",
      type: "closed",
      text: "Sono flessibile e mi adatto facilmente ai cambi di programma",
      scaleLabels: ["Per niente", "Molto"],
      options: [
        "Ho bisogno di pianificare tutto in anticipo",
        "Preferisco avere piani definiti",
        "Sono flessibile ma apprezzo un po' di struttura",
        "Mi adatto facilmente ai cambiamenti di programma",
        "Sono molto spontaneo, vivo alla giornata",
      ],
    },
    {
      id: "beh-8",
      type: "closed",
      text: "Accetto le critiche costruttive positivamente",
      scaleLabels: ["Per niente", "Molto"],
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
      text: "Mi piace fare progetti e pianificare il futuro",
      scaleLabels: ["Per niente", "Molto"],
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
      text: "How do you behave when you meet someone you're interested in?",
      template: "When I meet someone I'm interested in, {answer}",
      placeholder: "I approach slowly, I'm shy at first, I take initiative...",
    },
  ],
};

// ============================================
// TEST INFO
// ============================================

/** Current assessment name/version */
export const ASSESSMENT_NAME = "personality-v3";

export const SECTIONS: Section[] = ["psychological", "values", "interests", "behavioral"];

export function getQuestionById(id: string): Question | undefined {
  for (const section of SECTIONS) {
    const q = QUESTIONS[section].find(q => q.id === id);
    if (q) return q;
  }
  return undefined;
}

export const TOTAL_QUESTIONS = Object.values(QUESTIONS).flat().length;
