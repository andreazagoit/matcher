import { VALUES_OPTIONS, INTERESTS_OPTIONS } from "@/db/constants";

// Normalizza caratteri accentati per GraphQL enum
function normalizeForEnum(str: string): string {
  return str
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "") // Rimuove accenti
    .toUpperCase()
    .replace(/ /g, "_")
    .replace(/[^A-Z0-9_]/g, ""); // Solo lettere, numeri, underscore
}

// Genera enum GraphQL dai valori
const valuesEnum = VALUES_OPTIONS.map((v) => normalizeForEnum(v)).join("\n    ");
const interestsEnum = INTERESTS_OPTIONS.map((i) => normalizeForEnum(i)).join("\n    ");

export const userTypeDefs = `#graphql
  enum Value {
    ${valuesEnum}
  }

  enum Interest {
    ${interestsEnum}
  }

  type User {
    id: ID!
    firstName: String!
    lastName: String!
    email: String!
    birthDate: String!
    values: [Value!]!
    interests: [Interest!]!
    createdAt: String!
    updatedAt: String!
  }

  input CreateUserInput {
    firstName: String!
    lastName: String!
    email: String!
    birthDate: String!
    values: [Value!]!
    interests: [Interest!]!
  }

  input UpdateUserInput {
    firstName: String
    lastName: String
    email: String
    birthDate: String
    values: [Value!]
    interests: [Interest!]
  }
`;
