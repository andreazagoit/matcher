import { VALUES_OPTIONS } from "@/lib/models/values/operations";
import { INTERESTS_OPTIONS } from "@/lib/models/interests/operations";

// Genera enum GraphQL dai valori (già in formato standard)
const valuesEnum = VALUES_OPTIONS.join("\n    ");
const interestsEnum = INTERESTS_OPTIONS.join("\n    ");

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
    interests: [Value!]!
  }

  input UpdateUserInput {
    firstName: String
    lastName: String
    email: String
    birthDate: String
    values: [Value!]
    interests: [Interest!]
  }

  extend type Query {
    user(id: ID!): User
    users: [User!]!
    me: User
    findMatches(userId: ID!, limit: Int = 10): [User!]!
  }

  extend type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: ID!, input: UpdateUserInput!): User
    deleteUser(id: ID!): Boolean!
  }
`;

