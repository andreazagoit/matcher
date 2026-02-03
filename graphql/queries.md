# GraphQL Queries & Mutations

## 📝 Mutations

### 1. Creare un nuovo utente
```graphql
mutation CreateUser {
  createUser(
    input: {
      firstName: "Mario"
      lastName: "Rossi"
      email: "mario.rossi@example.com"
      birthDate: "1995-03-15"
      values: [FAMIGLIA, ONESTA, AMBIZIONE]
      interests: [SPORT, VIAGGI, CUCINA]
    }
  ) {
    id
    firstName
    lastName
    email
    birthDate
    values
    interests
    createdAt
  }
}
```

### 2. Aggiornare un utente
```graphql
mutation UpdateUser {
  updateUser(
    id: "USER_ID_QUI"
    input: {
      firstName: "Mario"
      interests: [SPORT, VIAGGI, CUCINA, FOTOGRAFIA]
    }
  ) {
    id
    firstName
    lastName
    email
    values
    interests
    updatedAt
  }
}
```

### 3. Eliminare un utente
```graphql
mutation DeleteUser {
  deleteUser(id: "USER_ID_QUI")
}
```

## 🔍 Queries

### 4. Ottenere un utente per ID
```graphql
query GetUser {
  user(id: "USER_ID_QUI") {
    id
    firstName
    lastName
    email
    birthDate
    values
    interests
    createdAt
    updatedAt
  }
}
```

### 5. Ottenere tutti gli utenti
```graphql
query GetAllUsers {
  users {
    id
    firstName
    lastName
    email
    birthDate
    values
    interests
    createdAt
  }
}
```

### 6. Ottenere l'utente corrente (me)
```graphql
query GetMe {
  me {
    id
    firstName
    lastName
    email
    birthDate
    values
    interests
  }
}
```

### 7. Cercare match per un utente (ANN Search)
```graphql
query FindMatches {
  findMatches(userId: "USER_ID_QUI", limit: 10) {
    id
    firstName
    lastName
    email
    birthDate
    values
    interests
    similarity
  }
}
```

## 📋 Valori Enum Disponibili

### Values (Valori)
```
FAMIGLIA
CARRIERA
AMICIZIA
AVVENTURA
STABILITA
CREATIVITA
SPIRITUALITA
SALUTE
LIBERTA
ONESTA
LEALTA
AMBIZIONE
EMPATIA
RISPETTO
CRESCITA_PERSONALE
```

### Interests (Interessi)
```
SPORT
MUSICA
VIAGGI
CUCINA
ARTE
CINEMA
LETTURA
FOTOGRAFIA
TECNOLOGIA
GAMING
NATURA
FITNESS
YOGA
DANZA
TEATRO
MODA
ANIMALI
VOLONTARIATO
ESCURSIONISMO
MEDITAZIONE
```

## 🎯 Esempi Completi

### Esempio 1: Creare utente e cercare match
```graphql
# Step 1: Crea utente
mutation {
  createUser(
    input: {
      firstName: "Alice"
      lastName: "Bianchi"
      email: "alice.bianchi@example.com"
      birthDate: "1996-05-20"
      values: [CREATIVITA, LIBERTA, EMPATIA]
      interests: [MUSICA, ARTE, YOGA]
    }
  ) {
    id
    firstName
    lastName
  }
}

# Step 2: Cerca match (usa l'id restituito)
query {
  findMatches(userId: "ID_RESTITUITO_DALLA_MUTATION", limit: 5) {
    id
    firstName
    lastName
    similarity
    values
    interests
  }
}
```

### Esempio 2: Query completa con tutti i campi
```graphql
query FullUserQuery {
  user(id: "USER_ID_QUI") {
    id
    firstName
    lastName
    email
    birthDate
    values
    interests
    createdAt
    updatedAt
  }
}
```

### Esempio 3: Match con similarity score
```graphql
query FindTopMatches {
  findMatches(userId: "USER_ID_QUI", limit: 20) {
    id
    firstName
    lastName
    email
    values
    interests
    similarity
  }
}
```

## 💡 Note

- Gli enum sono in **MAIUSCOLO** e senza accenti (es: `CREATIVITA` invece di `creatività`)
- Gli spazi sono sostituiti con underscore (es: `CRESCITA_PERSONALE`)
- Il campo `similarity` è disponibile solo nella query `findMatches`
- La `similarity` è un numero tra 0 e 1 (più alto = più simile)

