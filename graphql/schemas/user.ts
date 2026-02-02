import { VALUES_OPTIONS, INTERESTS_OPTIONS } from "@/db/constants";

// Genera enum GraphQL dai valori
const valuesEnum = VALUES_OPTIONS.map((v) =>
  v.toUpperCase().replace(/ /g, "_")
).join("\n    ");

const interestsEnum = INTERESTS_OPTIONS.map((i) =>
  i.toUpperCase().replace(/ /g, "_")
).join("\n    ");

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
    values: [Value!]
    interests: [Interest!]
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
