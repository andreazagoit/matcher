import { updateUserLocation, createUser } from "./operations";
import type { CreateUserInput } from "./validator";

const MILAN_CENTER = { lat: 45.4642, lon: 9.19 };

function randomMilanLocation() {
  const latJitter = (Math.random() - 0.5) * 0.14;
  const lonJitter = (Math.random() - 0.5) * 0.20;
  return { lat: MILAN_CENTER.lat + latJitter, lon: MILAN_CENTER.lon + lonJitter };
}

// ─── Photo pools ────────────────────────────────────────────────────────────

const MEN = [
  "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=800&q=80",
  "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=800&q=80",
  "https://images.unsplash.com/photo-1504257432389-52343af06ae3?w=800&q=80",
  "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=800&q=80",
  "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80",
  "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=800&q=80",
  "https://images.unsplash.com/photo-1463453091185-61582044d556?w=800&q=80",
  "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=800&q=80",
  "https://images.unsplash.com/photo-1480429370139-e0132c086e2a?w=800&q=80",
];

const WOMEN = [
  "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&q=80",
  "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=800&q=80",
  "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=800&q=80",
  "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=800&q=80",
  "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=800&q=80",
  "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=800&q=80",
  "https://images.unsplash.com/photo-1502764613149-7f1d229e230f?w=800&q=80",
  "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df?w=800&q=80",
  "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=800&q=80",
];

function p(pool: string[], ...indices: number[]) {
  return indices.map((i) => ({ type: "photo" as const, content: pool[i] }));
}

function pr(key: string, content: string) {
  return { type: "prompt" as const, promptKey: key, content };
}

// ─── Seed users ─────────────────────────────────────────────────────────────

export const SEED_USERS: CreateUserInput[] = [
  {
    username: "admin_system", name: "Admin System", email: "admin@matcher.local",
    birthdate: "1990-01-01", gender: "man",
    photos: p(MEN, 0, 1, 2, 3),
    prompts: [
      pr("best_quality", "Sono sempre puntuale. È un superpotere sottovalutato."),
      pr("core_value", "Onestà radicale, anche quando fa male."),
    ],
  },
  {
    username: "mario_rossi", name: "Mario Rossi", email: "mario.rossi@example.com",
    birthdate: "1995-03-15", gender: "man",
    sexualOrientation: ["straight"], heightCm: 180,
    relationshipIntent: ["serious_relationship"], relationshipStyle: "monogamous",
    hasChildren: "no", wantsChildren: "yes",
    smoking: "never", drinking: "sometimes", activityLevel: "active", religion: "none",
    jobTitle: "Ingegnere software", educationLevel: "master", schoolName: "Politecnico di Milano",
    languages: ["italian", "english"],
    photos: p(MEN, 1, 2, 3, 4),
    prompts: [
      pr("controversial_opinion", "Il caffè nel pomeriggio non è sbagliato. Sono pronto a difendere questa posizione."),
      pr("sunday_plan", "Mercato del sabato mattina, lettura al sole e cena improvvisata con quello che trovo in frigo."),
    ],
  },
  {
    username: "laura_bianchi", name: "Laura Bianchi", email: "laura.bianchi@example.com",
    birthdate: "1998-07-22", gender: "woman",
    sexualOrientation: ["bisexual"], heightCm: 165,
    relationshipIntent: ["serious_relationship", "casual_dating"], relationshipStyle: "ethical_non_monogamous",
    hasChildren: "no", wantsChildren: "open",
    smoking: "sometimes", drinking: "sometimes", activityLevel: "moderate", religion: "spiritual",
    jobTitle: "Grafica freelance", educationLevel: "bachelor", schoolName: "NABA Milano",
    languages: ["italian", "english", "french"],
    photos: p(WOMEN, 0, 1, 2, 3),
    prompts: [
      pr("cant_live_without", "Le playlist curate con cura. Una colonna sonora sbagliata può rovinare qualsiasi momento."),
      pr("win_me_over", "Proponi un piano spontaneo senza troppe aspettative. O preparami qualcosa da mangiare."),
    ],
  },
  {
    username: "alessandro_verdi", name: "Alessandro Verdi", email: "alessandro.verdi@example.com",
    birthdate: "1992-11-08", gender: "man",
    sexualOrientation: ["gay"], heightCm: 175,
    relationshipIntent: ["casual_dating"], relationshipStyle: "open",
    hasChildren: "no", wantsChildren: "no",
    smoking: "never", drinking: "regularly", activityLevel: "light", religion: "none",
    jobTitle: "Architetto", educationLevel: "master", schoolName: "Politecnico di Torino",
    languages: ["italian", "english", "spanish"],
    photos: p(MEN, 2, 3, 4, 5),
    prompts: [
      pr("innocent_red_flag", "Riorganizzare la libreria degli altri senza chiedere permesso."),
      pr("guilty_pleasure", "Guardare tutorial di cucina senza mai cucinare niente di quello che vedo."),
    ],
  },
  {
    username: "giulia_neri", name: "Giulia Neri", email: "giulia.neri@example.com",
    birthdate: "1996-05-30", gender: "woman",
    sexualOrientation: ["straight"], heightCm: 170,
    relationshipIntent: ["serious_relationship"], relationshipStyle: "monogamous",
    hasChildren: "no", wantsChildren: "yes",
    smoking: "never", drinking: "sometimes", activityLevel: "very_active", religion: "christian",
    jobTitle: "Medico", educationLevel: "phd", schoolName: "Università degli Studi di Bologna",
    languages: ["italian", "english"],
    photos: p(WOMEN, 1, 2, 3, 4),
    prompts: [
      pr("life_goal", "Aprire un ambulatorio in un paese di montagna. E avere un cane."),
      pr("love_language", "Atti di servizio — noto sempre quando qualcuno fa una cosa piccola senza che tu lo chieda."),
    ],
  },
  {
    username: "marco_ferrari", name: "Marco Ferrari", email: "marco.ferrari@example.com",
    birthdate: "1994-01-20", gender: "man",
    sexualOrientation: ["straight"], heightCm: 183,
    relationshipIntent: ["friendship"], relationshipStyle: "monogamous",
    hasChildren: "yes", wantsChildren: "no",
    smoking: "regularly", drinking: "regularly", activityLevel: "sedentary", religion: "none",
    jobTitle: "Commercialista", educationLevel: "bachelor",
    languages: ["italian"],
    photos: p(MEN, 3, 4, 5, 6),
    prompts: [
      pr("worst_habit", "Rispondere ai messaggi tre giorni dopo. Non è scortesia, è strategia."),
      pr("most_spontaneous", "Ho prenotato un volo per Lisbona il giorno prima. Non sapevo dove dormire."),
    ],
  },
  {
    username: "sofia_romano", name: "Sofia Romano", email: "sofia.romano@example.com",
    birthdate: "1997-09-12", gender: "woman",
    sexualOrientation: ["lesbian"], heightCm: 163,
    relationshipIntent: ["serious_relationship"], relationshipStyle: "monogamous",
    hasChildren: "no", wantsChildren: "yes",
    smoking: "never", drinking: "never", activityLevel: "active", religion: "none",
    jobTitle: "Insegnante", educationLevel: "master",
    languages: ["italian", "english", "german"],
    photos: p(WOMEN, 2, 3, 4, 5),
    prompts: [
      pr("perfect_first_date", "Una passeggiata lunga senza meta, poi trovare un posto a caso per mangiare."),
      pr("relationship_green_flag", "Saper stare in silenzio insieme senza imbarazzo."),
    ],
  },
  {
    username: "luca_colombo", name: "Luca Colombo", email: "luca.colombo@example.com",
    birthdate: "1993-06-25", gender: "man",
    sexualOrientation: ["straight"], heightCm: 178,
    relationshipIntent: ["casual_dating"], relationshipStyle: "open",
    hasChildren: "no", wantsChildren: "open",
    smoking: "sometimes", drinking: "sometimes", activityLevel: "moderate", religion: "none",
    jobTitle: "Product manager", educationLevel: "bachelor",
    languages: ["italian", "english"],
    photos: p(MEN, 4, 5, 6, 7),
    prompts: [
      pr("obsessed_with", "Mappe e itinerari. Ho una cartella con 40 trip mai fatti."),
      pr("two_truths_lie", "Ho vissuto a Berlino. Parlo giapponese di base. Ho finito Dark in un weekend."),
    ],
  },
  {
    username: "emma_ricci", name: "Emma Ricci", email: "emma.ricci@example.com",
    birthdate: "1999-02-14", gender: "woman",
    sexualOrientation: ["pansexual"], heightCm: 168,
    relationshipIntent: ["serious_relationship", "casual_dating"], relationshipStyle: "ethical_non_monogamous",
    hasChildren: "no", wantsChildren: "open",
    smoking: "never", drinking: "sometimes", activityLevel: "active", religion: "spiritual",
    jobTitle: "Fotografa", educationLevel: "bachelor",
    languages: ["italian", "english", "french"],
    photos: p(WOMEN, 3, 4, 5, 6),
    prompts: [
      pr("recent_discovery", "Le foto analogiche sviluppate male sono spesso più belle di quelle perfette."),
      pr("hidden_talent", "Riconosco le font a colpo d'occhio. È inutile ma mi fa sentire potente."),
    ],
  },
  {
    username: "andrea_marino", name: "Andrea Marino", email: "andrea.marino@example.com",
    birthdate: "1991-12-03", gender: "man",
    sexualOrientation: ["straight"], heightCm: 176,
    relationshipIntent: ["serious_relationship"], relationshipStyle: "monogamous",
    hasChildren: "yes", wantsChildren: "no",
    smoking: "never", drinking: "sometimes", activityLevel: "moderate", religion: "christian",
    jobTitle: "Avvocato", educationLevel: "master",
    languages: ["italian", "english"],
    photos: p(MEN, 5, 6, 7, 8),
    prompts: [
      pr("communication_style", "Diretto ma non brusco. Preferisco un conflitto onesto a una pace finta."),
      pr("bucket_list", "Fare il Cammino di Santiago. E leggere tutti i libri sulla mensola del soggiorno."),
    ],
  },
  {
    username: "chiara_greco", name: "Chiara Greco", email: "chiara.greco@example.com",
    birthdate: "1996-08-18", gender: "woman",
    sexualOrientation: ["straight"], heightCm: 167,
    relationshipIntent: ["casual_dating"], relationshipStyle: "monogamous",
    hasChildren: "no", wantsChildren: "yes",
    smoking: "never", drinking: "sometimes", activityLevel: "active", religion: "none",
    jobTitle: "UX designer", educationLevel: "bachelor",
    languages: ["italian", "english", "spanish"],
    photos: p(WOMEN, 4, 5, 6, 7),
    prompts: [
      pr("surprising_fact", "Ho studiato violino per 8 anni e ora non riesco a suonare neanche Happy Birthday."),
      pr("alter_ego", "Di notte divento quella che risponde ai DM con meme invece di parole."),
    ],
  },
  {
    username: "francesco_bruno", name: "Francesco Bruno", email: "francesco.bruno@example.com",
    birthdate: "1994-04-07", gender: "man",
    jobTitle: "Chef", educationLevel: "vocational",
    languages: ["italian", "english"],
    photos: p(MEN, 6, 7, 8, 0),
    prompts: [
      pr("controversial_opinion", "Il cibo fusion fatto male è peggio del cibo mediocre fatto bene."),
      pr("cant_live_without", "Un buon coltello da cucina e la musica mentre cucino."),
    ],
  },
  {
    username: "valentina_gallo", name: "Valentina Gallo", email: "valentina.gallo@example.com",
    birthdate: "1998-10-29", gender: "woman",
    jobTitle: "Biologa", educationLevel: "phd",
    languages: ["italian", "english", "german"],
    photos: p(WOMEN, 5, 6, 7, 8),
    prompts: [
      pr("i_am_always", "Quella che porta un libro in borsa anche solo per andare al bar sotto casa."),
      pr("dinner_with_anyone", "Carl Sagan. O mia nonna. Dipende dall'umore."),
    ],
  },
  {
    username: "matteo_conti", name: "Matteo Conti", email: "matteo.conti@example.com",
    birthdate: "1992-07-11", gender: "man",
    jobTitle: "Giornalista", educationLevel: "bachelor",
    languages: ["italian", "english"],
    photos: p(MEN, 7, 8, 0, 1),
    prompts: [
      pr("unpopular_take", "I podcast lunghi sono sopravvalutati. Dimmi la cosa importante in 10 minuti."),
      pr("dealbreaker", "Non leggere mai. Neanche un articolo, neanche per caso."),
    ],
  },
  {
    username: "alessia_costa", name: "Alessia Costa", email: "alessia.costa@example.com",
    birthdate: "1997-03-22", gender: "woman",
    jobTitle: "Ricercatrice", educationLevel: "phd",
    languages: ["italian", "english", "french"],
    photos: p(WOMEN, 6, 7, 8, 0),
    prompts: [
      pr("looking_for", "Qualcuno con cui avere conversazioni vere. Non solo \"come stai\" di cortesia."),
      pr("my_routine", "Sveglia presto, corsa, caffè lungo, tre ore di scrittura. Poi il caos."),
    ],
  },
  {
    username: "davide_fontana", name: "Davide Fontana", email: "davide.fontana@example.com",
    birthdate: "1995-11-05", gender: "man",
    jobTitle: "Sviluppatore mobile", educationLevel: "bachelor",
    languages: ["italian", "english"],
    photos: p(MEN, 8, 0, 1, 2),
    prompts: [
      pr("emoji_story", "🚀☕📱🎧🌙 — questa è la mia giornata tipo."),
      pr("sunday_plan", "Debug di un bug che ho introdotto venerdì alle 18. Poi aperitivo per dimenticare."),
    ],
  },
  {
    username: "martina_caruso", name: "Martina Caruso", email: "martina.caruso@example.com",
    birthdate: "1999-01-16", gender: "woman",
    jobTitle: "Studentessa universitaria", educationLevel: "bachelor",
    languages: ["italian", "english", "spanish"],
    photos: p(WOMEN, 7, 8, 0, 1),
    prompts: [
      pr("what_friends_say", "Che sono troppo diretta. Io preferisco chiamarla efficienza."),
      pr("win_me_over", "Portami a un posto che ami davvero, non quello che pensi mi piacerà."),
    ],
  },
  {
    username: "simone_mancini", name: "Simone Mancini", email: "simone.mancini@example.com",
    birthdate: "1993-09-28", gender: "man",
    jobTitle: "Consulente finanziario", educationLevel: "master",
    languages: ["italian", "english", "german"],
    photos: p(MEN, 0, 2, 4, 6),
    prompts: [
      pr("core_value", "La chiarezza è una forma di rispetto. Preferisco un no diretto a un forse infinito."),
      pr("bucket_list", "Vivere sei mesi all'estero senza agenda. Non per lavoro, solo per esistere altrove."),
    ],
  },
  {
    username: "giorgia_rizzo", name: "Giorgia Rizzo", email: "giorgia.rizzo@example.com",
    birthdate: "1996-12-09", gender: "woman",
    jobTitle: "Psicologa", educationLevel: "master",
    languages: ["italian", "english"],
    photos: p(WOMEN, 8, 0, 2, 4),
    prompts: [
      pr("love_language", "Ascolto attivo. Quando qualcuno ricorda una cosa piccola che hai detto settimane fa."),
      pr("controversial_opinion", "Il confine sano non è egoismo. È igiene relazionale."),
    ],
  },
  {
    username: "riccardo_lombardi", name: "Riccardo Lombardi", email: "riccardo.lombardi@example.com",
    birthdate: "1994-05-19", gender: "man",
    jobTitle: "Ingegnere civile", educationLevel: "master",
    languages: ["italian", "english"],
    photos: p(MEN, 1, 3, 5, 7),
    prompts: [
      pr("surprising_fact", "Ho costruito una barca a vela con mio padre. Non l'abbiamo mai usata."),
      pr("obsessed_with", "I ponti. Ogni città la guardo dal basso, dal livello dell'acqua."),
    ],
  },
  {
    username: "elisa_moretti", name: "Elisa Moretti", email: "elisa.moretti@example.com",
    birthdate: "1998-08-31", gender: "woman",
    jobTitle: "Social media manager", educationLevel: "bachelor",
    languages: ["italian", "english", "french"],
    photos: p(WOMEN, 1, 3, 5, 7),
    prompts: [
      pr("guilty_pleasure", "Guardare i reel di cani che non si aspettano la fine della canzone."),
      pr("alter_ego", "Online sono concisa e professionale. Nella vita reale parlo per venti minuti di una pianta."),
    ],
  },
  {
    username: "lorenzo_serra", name: "Lorenzo Serra", email: "lorenzo.serra@example.com",
    birthdate: "1993-02-14", gender: "man",
    jobTitle: "Fisioterapista", educationLevel: "bachelor",
    languages: ["italian", "english"],
    photos: p(MEN, 2, 4, 6, 8),
    prompts: [
      pr("best_quality", "Riesco a mettere a proprio agio anche le persone più chiuse. Lo dicono tutti."),
      pr("perfect_first_date", "Qualcosa di fisico — una camminata, un mercato, qualcosa da fare con le mani."),
    ],
  },
  {
    username: "francesca_de_luca", name: "Francesca De Luca", email: "francesca.deluca@example.com",
    birthdate: "1997-06-07", gender: "woman",
    jobTitle: "Traduttrice", educationLevel: "master",
    languages: ["italian", "english", "french", "spanish"],
    photos: p(WOMEN, 0, 2, 6, 8),
    prompts: [
      pr("i_am_always", "Quella che corregge mentalmente la grammatica. Ma non lo dico mai ad alta voce."),
      pr("recent_discovery", "In spagnolo esiste \"madrugada\" — quelle ore tra la notte e l'alba senza nome in italiano."),
    ],
  },
  {
    username: "tommaso_longo", name: "Tommaso Longo", email: "tommaso.longo@example.com",
    birthdate: "1995-10-23", gender: "man",
    jobTitle: "Musicista", educationLevel: "bachelor",
    languages: ["italian", "english"],
    photos: p(MEN, 3, 5, 7, 0),
    prompts: [
      pr("hidden_talent", "Suono quattro strumenti ma non riesco ad arrivare puntuale da nessuna parte."),
      pr("two_truths_lie", "Ho suonato all'apertura di un concerto di Brunori Sas. Ho un gatto di nome Coltrane. Non ho mai sentito i Beatles per intero."),
    ],
  },
  {
    username: "beatrice_leone", name: "Beatrice Leone", email: "beatrice.leone@example.com",
    birthdate: "1998-04-11", gender: "woman",
    jobTitle: "Stylist", educationLevel: "vocational",
    languages: ["italian", "english", "french"],
    photos: p(WOMEN, 3, 5, 7, 1),
    prompts: [
      pr("what_friends_say", "Che vesto meglio io i loro vestiti di loro. È un complimento, credo."),
      pr("innocent_red_flag", "Giudicare le persone in base alle scarpe che portano. Funziona sempre."),
    ],
  },
  {
    username: "gabriele_martinelli", name: "Gabriele Martinelli", email: "gabriele.martinelli@example.com",
    birthdate: "1992-12-28", gender: "man",
    jobTitle: "Veterinario", educationLevel: "phd",
    languages: ["italian", "english"],
    photos: p(MEN, 4, 6, 8, 1),
    prompts: [
      pr("dinner_with_anyone", "Jane Goodall. Le chiederei se gli animali ci trovano ridicoli."),
      pr("cant_live_without", "Il mio cane Argo. E il silenzio della mattina presto."),
    ],
  },
];

// ─── Seed function ───────────────────────────────────────────────────────────

export async function seedUsers() {
  console.log(`\n👤 Seeding ${SEED_USERS.length} users...`);

  const created: Record<string, { id: string; email: string }> = {};

  for (const seed of SEED_USERS) {
    const loc = randomMilanLocation();
    const user = await createUser(seed);
    await updateUserLocation(user.id, loc.lat, loc.lon);
    created[user.email] = user;
    console.log(`  ✓ ${seed.name}`);
  }

  console.log(`  → ${Object.keys(created).length} users created`);
  return created;
}
