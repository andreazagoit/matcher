/**
 * GraphQL Schema per Users
 * 
 * NUOVA ARCHITETTURA:
 * - User: solo dati anagrafici base
 * - Profile: profilo con traits + embeddings (per matching)
 * - Test: sessioni e risposte ai questionari
 * 
 * Values/Interests ora fanno parte del sistema test, non più dell'utente direttamente.
 */

export const userTypeDefs = `#graphql
  """
  Utente base - dati anagrafici
  """
  type User {
    id: ID!
    firstName: String!
    lastName: String!
    email: String!
    birthDate: String!
    gender: Gender
    createdAt: String!
    updatedAt: String!
    
    # Relazione con profilo (opzionale, esiste dopo test completato)
    profile: Profile
  }

  """
  Profilo utente con traits e dati per matching
  """
  type Profile {
    id: ID!
    userId: ID!
    
    # Traits aggregati per asse
    psychologicalTraits: JSON
    valuesTraits: JSON
    interestsTraits: JSON
    behavioralTraits: JSON
    
    # Descrizioni testuali generate
    psychologicalDescription: String
    valuesDescription: String
    interestsDescription: String
    behavioralDescription: String
    
    # Timestamps
    createdAt: String!
    updatedAt: String!
    embeddingsComputedAt: String
  }


  input CreateUserInput {
    firstName: String!
    lastName: String!
    email: String!
    birthDate: String!
    gender: Gender
  }

  input UpdateUserInput {
    firstName: String
    lastName: String
    email: String
    birthDate: String
    gender: Gender
  }

  enum Gender {
    man
    woman
    non_binary
  }

  input MatchOptions {
    limit: Int = 10
    gender: [Gender!]
    minAge: Int
    maxAge: Int
  }

  extend type Query {
    user(id: ID!): User
    users: [User!]!
    me: User
    
    # Matching (richiede profilo completato)
    # Se userId non è fornito, usa l'utente autenticato
    findMatches(userId: ID, options: MatchOptions): [User!]!
  }

  extend type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: ID!, input: UpdateUserInput!): User
    deleteUser(id: ID!): Boolean!
  }
`;

